__kernel void RispDynamicsFwdKernel(
    __global const float* x,
    __global const float* v_thresh,
    __global const float* weights,
    __global const int* delays,
    __global const int* incoming,
    __global const int* incoming_ids,
    float v_decay,
    float v_rest,
    __global const int* is_input_neuron,
    __global float* v,
    __global long* s,
    __global float* v_pre,
    int num_neurons,
    int num_steps,
    int timestep,
    int max_incoming) {

    unsigned neuron_id = get_global_id(0);
    if (neuron_id >= (unsigned)num_neurons) {
        return;
    }

    float V_thresh = v_thresh[neuron_id];
    float V_rest = v_rest;
    float V = v[neuron_id];
    int idx = neuron_id * num_steps + timestep;

    // Binary leak (v_decay of 1 -> leak_mode "all")
    if (v_decay > 0.0f) {
        V = 0.0f;
    }

    // Min potential reset 
    if (V < V_rest) {
        V = V_rest;
    }

    // Grab input spikes from x
    if (is_input_neuron[neuron_id]) {
        V += x[idx];
    }

    // Check incoming synapses for spikes
    for (int i = 0; i < incoming[neuron_id]; i++) {
        int incoming_id = incoming_ids[neuron_id * max_incoming + i];
        int source_ts = timestep - delays[neuron_id * max_incoming + i];
        float weight = weights[neuron_id * max_incoming + i];

        if (source_ts < 0) {
            continue;
        }

        float source_spike = s[incoming_id * num_steps + source_ts];
        V += source_spike * weight;
    }

    v_pre[idx] = V;

    const bool has_spiked = V >= V_thresh;

    // Reset charge when we spike
    if (has_spiked) {
        V = 0.0f;
    }
    // Final Min potential reset
    if (V < V_rest) {
        V = V_rest;
    }

    v[neuron_id] = V; // Charge (for next timestep)
    s[idx] = (long)has_spiked; // Spike
}

__kernel void RispSpikeLoss(
    __global float* dL_ds,
    __global int* correct,
    __global float* loss,
    __global const long* s,
    int num_neurons,
    int num_output_neurons,
    int num_steps,
    int target_idx
) {
    unsigned id = get_global_id(0);
    // FIXME: serial operation for now
    if (id > 0) {
        return;
    }

    float sum = 0.0;
    int max_idx = 0;
    float max_val = 0;
    for (int i = 0; i < num_output_neurons; i++) {
        int base_idx = (i + (num_neurons - num_output_neurons)) * num_steps;

        for (int t = 0; t < num_steps; t++) {
            dL_ds[i] += s[base_idx + t];
        }

        if (dL_ds[i] > max_val) {
            max_val = dL_ds[i];
            max_idx = i;
        }
        
        dL_ds[i] /= (float)num_steps;
        sum += exp(dL_ds[i]);
    }

    for (int i = 0; i < num_output_neurons; i++) {
        dL_ds[i] /= sum;
    }

    *correct += max_idx == target_idx;
    *loss -= log(dL_ds[target_idx]);

    for (int i = 0; i < num_output_neurons; i++) {
        dL_ds[i] -= (i == target_idx ? 1.0f : 0.0f);
    }

}

__kernel void RispDynamicsBwdKernel(
    __global const float* dL_ds,
    __global const long* s,
    __global const float* v_pre,
    __global const float* v_thresh,
    __global const int* is_output_neuron,
    __global float* spike_grad_history,   // num_neurons * num_steps
    __global float* voltage_grad_history, // num_neurons * num_steps
    __global float* future_mem_grad, // num_neurons
    __global float* delta_W, // num_neurons * max_incoming
    __global const float* weights, // num_neurons * max_incoming
    __global const int* delays, // num_neurons * max_incoming
    __global const int* incoming, // num_neurons 
    __global const int* incoming_ids, // num_neurons * max_incoming
    float v_decay,
    float v_rest,
    float tau_rho,
    float scale_rho,
    int num_neurons,
    int num_output_neurons,
    int num_steps,
    int timestep,
    int max_incoming) {

    unsigned neuron_id = get_global_id(0);
    if (neuron_id >= (unsigned)num_neurons) {
        return;
    }

    int base = neuron_id * num_steps;

    // leak_mode "all" == 1.0
    const bool has_decay = (v_decay > 0.0f);
    const float tau_rho_scaled = tau_rho;

    int idx = base + timestep;

    if (is_output_neuron[neuron_id]) {
        spike_grad_history[idx] +=
            dL_ds[neuron_id - num_output_neurons]
            / (float)num_steps;
    }

    float dL_dV = voltage_grad_history[idx] + future_mem_grad[neuron_id];

    const float v_pre_t = v_pre[idx];
    const float dy_t_dV_post = 1.0f;
    const float dV_post_dV_pre = 1.0f - (s[idx] > 0 ? 1.0f : 0.0f);
    const float dV_pre_dx_t = 1.0;
    const float dV_post_ds_t = (v_rest > 0) ? (v_rest - v_pre_t) : (-v_pre_t);

    const float ds_t_dV_pre = ((scale_rho / (2.0f * tau_rho_scaled)) * exp(-fabs(v_pre_t - v_thresh[neuron_id]) / tau_rho_scaled));

    const float grad = (dL_dV * dV_post_dV_pre * dV_pre_dx_t) +
        (dL_dV * dV_post_ds_t * ds_t_dV_pre * dV_pre_dx_t) +
        (spike_grad_history[idx] * ds_t_dV_pre * dV_pre_dx_t);

    if (timestep > 0) {
        const float dV_pre_dV_leak = 1.0f;
        const float dV_leak_dV_t1 = (v_pre_t >= v_rest) ? (1.0f - v_decay) : 0.0f;
        future_mem_grad[neuron_id] = (dL_dV * dV_post_dV_pre * dV_pre_dV_leak * dV_leak_dV_t1) +
            (dL_dV * dV_post_ds_t * ds_t_dV_pre * dV_pre_dV_leak * dV_leak_dV_t1) +
            (spike_grad_history[idx] * ds_t_dV_pre * dV_pre_dV_leak * dV_leak_dV_t1);
    }

    for (int i = 0; i < incoming[neuron_id]; i++) {
        int incoming_id = incoming_ids[neuron_id * max_incoming + i];
        int source_ts = timestep - delays[neuron_id * max_incoming + i];
        float source_spike = s[incoming_id * num_steps + source_ts];
        float weight = weights[neuron_id * max_incoming + i];

        if (source_ts < 0) {
            continue;
        }

        delta_W[neuron_id * max_incoming + i] += source_spike * grad;
        spike_grad_history[incoming_id * num_steps + source_ts] +=
            grad * weight;
    }
}

__kernel void RispWeightUpdates(
    __global float* m_weights, // num_neurons * max_incoming
    __global float* v_weights, // num_neurons * max_incoming
    __global const float* delta_W, // num_neurons * max_incoming
    __global float* weights, // num_neurons * max_incoming
    __global const int* incoming, // num_neurons 
    int num_neurons,
    int max_incoming,
    float learning_rate,
    float decay_rate,
    int current_batch_size,
    int batch_size,
    int batch_start,
    int epoch,
    float beta1,
    float beta2,
    float b1_t,
    float b2_t,
    int timesteps,
    int num_observations) {

    unsigned neuron_id = get_global_id(0);
    if (neuron_id >= (unsigned)num_neurons) {
        return;
    }

    float inv_batch = 1.0f / ((float)current_batch_size * timesteps);

    for (int i = 0; i < incoming[neuron_id]; i++) {
        const int idx = neuron_id * max_incoming + i;
        
        const float delta = delta_W[idx] * inv_batch;
        m_weights[idx] = beta1 * m_weights[idx] + (1.0 - beta1) * delta;
        v_weights[idx] = beta2 * v_weights[idx] + (1.0 - beta2) * (delta * delta);

        const float mW_hat = m_weights[idx] / (1.0 - b1_t);
        const float vW_hat = v_weights[idx] / (1.0 - b2_t);

        float lr;
        if (epoch == 0) {
            lr = ((batch_size + batch_size) / num_observations) * learning_rate;
        } else {
            lr = learning_rate;
        }

        float weight = weights[idx];
        weight -= lr * mW_hat / (sqrt(vW_hat + 1.0e-8f));
        weight -= lr * decay_rate * weight;

        weights[idx] = weight;
    }
}

__kernel void RispEncodeInputs(
    __global float* x, // num_neurons * timesteps
    __global const float* data,
    int cols,
    int num_input_neurons,
    int timesteps,
    int observation_idx,
    float spike_value_factor) {
    // TODO handle timeseries

    unsigned neuron_id = get_global_id(0);
    if (neuron_id >= (unsigned)num_input_neurons) {
        return;
    }

    float val = data[observation_idx * (cols*2) + neuron_id];
    if (val <= 0.0f) {
        return;
    }

    int base_idx = neuron_id * timesteps;
    // x[base_idx] = val;
    // x[base_idx+1] = observation_idx * (cols*2) + neuron_id; 
    for (float i = 0.0f; i < (float)timesteps; i += val) {
        x[base_idx + (int)i] = spike_value_factor;
    }
}
