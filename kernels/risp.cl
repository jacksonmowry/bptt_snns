__kernel void RispEncodeInputs(__global short* x, // num_neurons * timesteps
                               __global const double* data, int cols,
                               int num_input_neurons, int timesteps,
                               uint observation_idx, short spike_value_factor) {
    // TODO handle timeseries

    unsigned neuron_id = get_global_id(0);
    if (neuron_id >= (unsigned)num_input_neurons) {
        return;
    }

    double val = data[observation_idx * (cols * 2) + neuron_id];
    if (val <= 0.0) {
        return;
    }

    ushort base_idx = neuron_id * timesteps;
    for (double i = 0.0; i < (double)timesteps; i += val) {
        x[base_idx + (short)i] = spike_value_factor;
    }
}

__kernel void RispDynamicsFwdKernel(
    __global const short* x, __global const short* v_thresh,
    __global const short* weights, __global const ushort* delays,
    __global const ushort* incoming, __global const ushort* incoming_ids,
    short v_decay, short v_rest, __global const uchar* is_input_neuron,
    __global short* v, // Per-neuron
    __global char* s, __global short* v_pre, ushort num_neurons,
    ushort num_steps, ushort timestep, ushort max_incoming) {

    ushort neuron_id = get_global_id(0);
    if (neuron_id >= (unsigned)num_neurons) {
        return;
    }

    short V_thresh = v_thresh[neuron_id];
    short V_rest   = v_rest;
    short V        = timestep == 0 ? 0 : v[neuron_id];
    ushort idx     = neuron_id * num_steps + timestep;

    // Binary leak (v_decay of 1 -> leak_mode "all")
    if (v_decay > 0) {
        V = 0;
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
        const ushort incoming_id = incoming_ids[neuron_id * max_incoming + i];
        const short weight       = weights[neuron_id * max_incoming + i];
        const short source_ts =
            (short)timestep - (short)delays[neuron_id * max_incoming + i];

        if (source_ts < 0) {
            continue;
        }

        const char source_spike = s[incoming_id * num_steps + source_ts];
        V += source_spike ? weight : 0;
    }

    v_pre[idx] = V;

    const bool has_spiked = V >= V_thresh;

    // Reset charge when we spike
    if (has_spiked) {
        V = 0;
    }
    // Final Min potential reset
    if (V < V_rest) {
        V = V_rest;
    }

    v[neuron_id] = V;                // Charge (for next timestep)
    s[idx]       = (char)has_spiked; // Spike
}

__kernel void RispSpikeLoss(__global double* dL_ds, __global double* correct,
                            __global double* loss, __global const char* s,
                            ushort num_neurons, ushort num_output_neurons,
                            ushort num_steps, ushort target_idx) {
    unsigned id = get_global_id(0);
    // FIXME: serial operation for now
    if (id > 0) {
        return;
    }

    double sum     = 0.0;
    int max_idx   = 0;
    double max_val = 0;

    for (int i = 0; i < num_output_neurons; i++) {
        int base_idx = (i + (num_neurons - num_output_neurons)) * num_steps;

        for (int t = 0; t < num_steps; t++) {
            dL_ds[i] += s[base_idx + t];
        }

        dL_ds[i] /= (double)num_steps;
        if (dL_ds[i] > max_val) {
            max_val = dL_ds[i];
            max_idx = i;
        }
    }

    for (int i = 0; i < num_output_neurons; i++) {
        dL_ds[i] = exp(dL_ds[i] - max_val);
        sum += dL_ds[i];
    }

    for (int i = 0; i < num_output_neurons; i++) {
        dL_ds[i] /= sum;
    }

    *correct += max_idx == target_idx;
    *loss -= log(dL_ds[target_idx] + 1e-8f);

    for (int i = 0; i < num_output_neurons; i++) {
        dL_ds[i] -= (i == target_idx ? 1.0 : 0.0);
    }
}

typedef struct {
    short v_decay;
    short v_rest;
    double tau_rho;
    double scale_rho;
    ushort num_neurons;
    ushort num_output_neurons;
    short num_steps;
    ushort max_incoming;
    double scale_factor;
} BwdParams;

__kernel void RispDynamicsBwdKernel(
    __global const double* dL_ds, __global const char* s,
    __global const short* v_pre, __global const short* v_thresh,
    __global const uchar* is_output_neuron,
    __global double* spike_grad_history,   // num_neurons * num_steps
    __global double* voltage_grad_history, // num_neurons * num_steps
    __global double* future_mem_grad,      // num_neurons
    __global double* delta_W,              // num_neurons * max_incoming
    __global const short* weights,        // num_neurons * max_incoming
    __global const ushort* delays,        // num_neurons * max_incoming
    __global const ushort* incoming,      // num_neurons
    __global const ushort* incoming_ids,  // num_neurons * max_incoming
    __constant BwdParams* params,
    short timestep
) {
    
    unsigned neuron_id = get_global_id(0);
    if (neuron_id >= (unsigned)params->num_neurons) {
        return;
    }

    int base = neuron_id * params->num_steps;
    // leak_mode "all" == 1.0
    const bool has_decay       = (params->v_decay > 0);
    const double tau_rho_scaled = params->tau_rho;

    int idx = base + timestep;

    if (is_output_neuron[neuron_id]) {
        spike_grad_history[idx] +=
            dL_ds[neuron_id - (params->num_neurons - params->num_output_neurons)] / (double)params->num_steps;
    }

    double dL_dV = voltage_grad_history[idx] + future_mem_grad[neuron_id];

    const double v_pre_t        = v_pre[idx] * params->scale_factor;
    const double dy_t_dV_post   = 1.0;
    const double dV_post_dV_pre = 1.0 - (s[idx] > 0 ? 1.0 : 0.0);
    const double dV_pre_dx_t    = 1.0;
    const double dV_post_ds_t   = ((params->v_rest * params->scale_factor) > 0)
                                     ? ((params->v_rest * params->scale_factor) - v_pre_t)
                                     : (-v_pre_t);

    const double ds_t_dV_pre =
        ((params->scale_rho / (2.0 * tau_rho_scaled)) *
         exp(-fabs(v_pre_t - (v_thresh[neuron_id] * params->scale_factor)) /
             tau_rho_scaled));

    const double grad = (dL_dV * dV_post_dV_pre * dV_pre_dx_t) +
                       (dL_dV * dV_post_ds_t * ds_t_dV_pre * dV_pre_dx_t) +
                       (spike_grad_history[idx] * ds_t_dV_pre * dV_pre_dx_t);

    if (timestep > 0) {
        const double dV_pre_dV_leak = 1.0;
        const double dV_leak_dV_t1 =
            (v_pre_t >= (params->v_rest * params->scale_factor)) ? (1.0 - params->v_decay) : 0.0;
        future_mem_grad[neuron_id] =
            (dL_dV * dV_post_dV_pre * dV_pre_dV_leak * dV_leak_dV_t1) +
            (dL_dV * dV_post_ds_t * ds_t_dV_pre * dV_pre_dV_leak *
             dV_leak_dV_t1) +
            (spike_grad_history[idx] * ds_t_dV_pre * dV_pre_dV_leak *
             dV_leak_dV_t1);
    }

    for (int i = 0; i < incoming[neuron_id]; i++) {
        ushort incoming_id = incoming_ids[neuron_id * params->max_incoming + i];
        short source_ts    = timestep - delays[neuron_id * params->max_incoming + i];
        double weight = weights[neuron_id * params->max_incoming + i] * params->scale_factor;

        if (source_ts < 0) {
            continue;
        }

        bool source_spike  = s[incoming_id * params->num_steps + source_ts];

        delta_W[neuron_id * params->max_incoming + i] += source_spike * grad;
        spike_grad_history[incoming_id * params->num_steps + source_ts] +=
            grad * weight;
    }
}

inline double quantize(double weight, int steps) {
    int x = round(weight / (2.0 / steps));

    return x * (2.0 / steps);
}

// Spawned as 2D kernel
__kernel void RispWeightUpdates(
    __global double* m_weights, // num_neurons * max_incoming
    __global double* v_weights, // num_neurons * max_incoming
    __global double* delta_W, // num_neurons * max_incoming
    __global short* weights, // num_neurons * max_incoming
    __global const int* incoming, // num_neurons 
    ushort num_neurons,
    ushort max_incoming,
    double learning_rate,
    double decay_rate,
    ushort current_batch_size,
    ushort batch_size,
    uint batch_start,
    uint epoch,
    double beta1,
    double beta2,
    double b1_t,
    double b2_t,
    ushort timesteps,
    uint num_observations,
    double scale_factor, short _min, short _max, int steps) {

    unsigned global_id = get_global_id(0);
    if (global_id >= (unsigned)num_neurons * max_incoming) {
        return;
    }

    unsigned neuron_id = global_id / max_incoming;
    unsigned synapse_id = global_id % max_incoming;
    if (synapse_id >= incoming[neuron_id]) {
        return;
    }

    double inv_batch = 1.0 / ((double)current_batch_size * timesteps);

    const double delta = delta_W[global_id] * inv_batch;
    delta_W[global_id] = 0.0;
    double prev_m_weight = m_weights[global_id];
    double prev_v_weight = v_weights[global_id];
    double new_m_weight = beta1 * prev_m_weight + (1.0 - beta1) * delta;
    double new_v_weight = beta2 * prev_v_weight + (1.0 - beta2) * (delta * delta);

    const double mW_hat = new_m_weight / (1.0 - b1_t);
    const double vW_hat = new_v_weight / (1.0 - b2_t);

    m_weights[global_id] = new_m_weight;
    v_weights[global_id] = new_v_weight;

    double lr;
    if (false && epoch == 0) {
        lr = ((batch_size + batch_size) / num_observations) * learning_rate;
    } else {
        lr = learning_rate;
    }

    short old_weight = weights[global_id];
    double weight = weights[global_id] * scale_factor;
    weight -= lr * mW_hat / (sqrt(vW_hat + 1.0e-8));
    weight -= lr * decay_rate * weight;
    weight = clamp(weight, -1.0, 1.0);
    weight = quantize(weight, steps);

    short new_weight = (short)(weight / scale_factor);
    weights[global_id] = clamp(new_weight, _min, _max);
}
