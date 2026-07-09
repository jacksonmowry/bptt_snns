#include "kernel.hpp"

string opencl_c_container() {
    return R( // ####### Begining of OpenCL C code########

        kernel void risp_encode_inputs_kernel(
            global short* x, // num_neurons * timesteps
            global const double* data, int cols, int num_input_neurons,
            int timesteps, uint observation_idx, short spike_value_factor) {
            const uint neuron_id = get_global_id(0);
            if (neuron_id >= (uint)num_input_neurons) {
                return;
            }

            double val = data[observation_idx * (cols * 2) + neuron_id];
            if (val <= 0.0) {
                return;
            }

            uint base_idx = neuron_id * timesteps;
            for (double i = 0.0f; i < (double)timesteps; i += val) {
                x[base_idx + (uint)i] = spike_value_factor;
            }
        }

        kernel void risp_forward_kernel(
            global const short* x, global const short* v_thresh,
            global const short* weights, global const uint* delays,
            global const uint* incoming, global const uint* incoming_ids,
            global const uchar* is_input_neuron, global short* v,
            global char* s, global short* v_pre, short v_decay, short v_rest,
            uint num_neurons, uint num_steps, uint timestep,
            uint max_incoming) {
            const uint neuron_id = get_global_id(0);
            if (neuron_id >= (uint)num_neurons) {
                return;
            }

            short V_thresh = v_thresh[neuron_id];
            short V_rest   = v_rest;
            short V        = timestep == 0 ? 0 : v[neuron_id];
            uint idx     = neuron_id * num_steps + timestep;
            bool has_event = false;

            short total_input = 0;
            if (is_input_neuron[neuron_id]) {
                const short input_spike = x[idx];
                if (input_spike != 0) {
                    has_event = true;
                }
                total_input += input_spike;
            }

            // Check presynaptic neurons for spikes
            for (int i = 0; i < incoming[neuron_id]; i++) {
                const uint incoming_id =
                    incoming_ids[neuron_id * max_incoming + i];
                const short weight = weights[neuron_id * max_incoming + i];
                const short source_ts =
                    (short)timestep -
                    (short)delays[neuron_id * max_incoming + i];

                if (source_ts < 0) {
                    continue;
                }

                const char source_spike =
                    s[incoming_id * num_steps + source_ts];
                if (source_spike) {
                    has_event = true;
                    total_input += weight;
                }
            }

            // Match CPU/event-driven behavior: leak/min-potential only apply
            // when an event arrives for this neuron.
            if (has_event) {
                if (v_decay > 0) {
                    V = 0;
                }

                if (V < V_rest) {
                    V = V_rest;
                }

                V += total_input;
            }

            v_pre[idx] = V;

            const bool has_spiked = has_event && (V >= V_thresh);

            // Reset charge when we spike
            if (has_spiked) {
                V = 0;
            }

            v[neuron_id] = V;
            s[idx]       = (char)has_spiked;
        }

        kernel void risp_loss_kernel(
            global const char* s, global float* dL_ds, global float* correct,
            global float* loss, uint num_neurons, uint num_output_neurons,
            uint num_steps, uint target_idx) {
            const uint neuron_id = get_global_id(0);
            // Serial for now
            if (neuron_id > 0) {
                return;
            }

            float sum     = 0.0f;
            int max_idx   = 0;
            float max_val = 0.0f;

            for (int i = 0; i < num_output_neurons; i++) {
                int base_idx =
                    (i + (num_neurons - num_output_neurons)) * num_steps;

                for (int t = 0; t < num_steps; t++) {
                    dL_ds[i] += s[base_idx + t];
                }

                dL_ds[i] /= (float)num_steps;
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
                dL_ds[i] -= (i == target_idx ? 1.0f : 0.0f);
            }
        }

        // Kernel 1: per-neuron gradient computation.
        // Computes neuron_grad and updates future_mem_grad.
        kernel void risp_backward_grad_kernel(
            global const float* dL_ds, global const char* s,
            global const short* v_pre, global const short* v_thresh,
            global const uchar* is_output_neuron,
            global float* spike_grad_history,
            global float* voltage_grad_history, global float* future_mem_grad,
            global float* neuron_grad, short v_decay, float v_rest,
            float tau_rho, float scale_rho, uint num_neurons,
            uint num_output_neurons, short num_steps, float scale_factor,
            short timestep) {
            const uint neuron_id = get_global_id(0);
            if (neuron_id >= (uint)num_neurons) {
                return;
            }

            const int base              = neuron_id * num_steps;
            const int idx               = base + timestep;
            const float tau_rho_scaled  = tau_rho;

            // Sum spike_grad_history for this neuron at this timestep.
            float spike_grad_sum = spike_grad_history[idx];
            if (is_output_neuron[neuron_id]) {
                spike_grad_sum +=
                    dL_ds[neuron_id - (num_neurons - num_output_neurons)] /
                    (float)num_steps;
            }

            const float dL_dV =
                voltage_grad_history[idx] + future_mem_grad[neuron_id];
            const float v_pre_t        = v_pre[idx] * scale_factor;
            const float dV_post_dV_pre = 1.0f - (s[idx] > 0 ? 1.0f : 0.0f);
            const float dV_post_ds_t =
                ((v_rest) > 0) ? ((v_rest) - v_pre_t) : (-v_pre_t);
            const float ds_t_dV_pre =
                ((scale_rho / (2.0f * tau_rho_scaled)) *
                 exp(-fabs(v_pre_t - (v_thresh[neuron_id] * scale_factor)) /
                     tau_rho_scaled));

            const float grad =
                (dL_dV * dV_post_dV_pre) +
                (dL_dV * dV_post_ds_t * ds_t_dV_pre) +
                (spike_grad_sum * ds_t_dV_pre);

            neuron_grad[neuron_id] = grad;

            if (timestep > 0) {
                const float dV_leak_dV_t1 =
                    (v_pre_t >= (v_rest)) ? (1.0f - (float)v_decay) : 0.0f;

                future_mem_grad[neuron_id] =
                    (dL_dV * dV_post_dV_pre * dV_leak_dV_t1) +
                    (dL_dV * dV_post_ds_t * ds_t_dV_pre *
                     dV_leak_dV_t1) +
                    (spike_grad_sum * ds_t_dV_pre *
                     dV_leak_dV_t1);
            }
        }

        // Kernel 2: per-synapse weight gradient accumulation.
        // Uses neuron_grad to update delta_W and spike_grad_history.
        kernel void risp_backward_delta_w_kernel(
            global const float* neuron_grad, global const char* s,
            global const short* weights, global const uint* delays,
            global const uint* incoming, global const uint* incoming_ids,
            global float* spike_grad_history, global float* delta_W,
            uint num_neurons, uint max_incoming, short num_steps,
            float scale_factor, short timestep) {
            const uint global_id = get_global_id(0);

            const uint neuron_id  = global_id / max_incoming;
            const uint synapse_id = global_id % max_incoming;
            if (neuron_id >= num_neurons ||
                synapse_id >= incoming[neuron_id]) {
                return;
            }

            uint incoming_id = incoming_ids[global_id];
            short source_ts    = timestep - (short)delays[global_id];
            float weight       = weights[global_id] * scale_factor;

            if (source_ts < 0) {
                return;
            }

            const char source_spike =
                s[incoming_id * num_steps + source_ts];
            const float grad = neuron_grad[neuron_id];

            delta_W[global_id] += source_spike * grad;
            atomic_fetch_add(
                (atomic_float*)&spike_grad_history[incoming_id * num_steps +
                                                   source_ts],
                (grad * weight));
        }

        kernel void weight_updates_kernel(
            global const uint* incoming, global float* m_weights,
            global float* v_weights, global float* delta_W,
            global short* weights, uint num_neurons, uint max_incoming,
            float learning_rate, float decay_rate, uint current_batch_size,
            uint batch_size, uint batch_start, uint epoch, float beta1,
            float beta2, float b1_t, float b2_t, uint timesteps,
            uint num_observations, float scale_factor, short min_weight,
            short max_weight, int steps) {
            const uint global_id  = get_global_id(0);
            const uint neuron_id  = global_id / max_incoming;
            const uint synapse_id = global_id % max_incoming;
            if (global_id >= num_neurons * max_incoming ||
                neuron_id >= num_neurons || synapse_id >= incoming[neuron_id]) {
                return;
            }

            const float inv_batch =
                1.0f / ((float)current_batch_size * (float)timesteps);
            const float delta = delta_W[global_id] * inv_batch;
            delta_W[global_id] = 0.0f;

            const float new_m_weight = m_weights[global_id] =
                beta1 * m_weights[global_id] + (1.0f - beta1) * delta;
            const float new_v_weight = v_weights[global_id] =
                beta2 * v_weights[global_id] + (1.0f - beta2) * (delta * delta);

            const float mW_hat = new_m_weight / (1.0f - b1_t);
            const float vW_hat = new_v_weight / (1.0f - b2_t);

            float lr;
            if (epoch == 0) {
                lr = ((batch_start + batch_size) /
                      (float)num_observations) *
                    learning_rate;
            } else {
                lr = learning_rate;
            }

            float weight = weights[global_id] * scale_factor;
            weight -= lr * mW_hat / sqrt(vW_hat + 1.0e-8f);
            weight -= lr * decay_rate * weight;
            weight = clamp(weight, -1.0f, 1.0f);
            weight = round(weight / scale_factor);

            weights[global_id] = clamp((short)(weight), min_weight, max_weight);
        }

    );
} // ####### End of OpenCL C coe
