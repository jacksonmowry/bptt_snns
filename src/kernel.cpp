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

            const double val = data[observation_idx * (cols * 2) + neuron_id];
            if (val <= 0.0) {
                return;
            }

            const uint base_idx = neuron_id * timesteps;
            for (double i = 0.0; i < (double)timesteps; i += val) {
                x[base_idx + (uint)i] = spike_value_factor;
            }
        }

        kernel void risp_encode_timeseries_inputs_kernel(
            global short* x, // num_neurons * timesteps
            global const double* data, int dataset_timesteps,
            int num_input_neurons, int timesteps, uint observation_idx,
            short spike_value_factor) {
            const uint neuron_id = get_global_id(0);
            if (neuron_id >= (uint)num_input_neurons) {
                return;
            }

            const int encoding_window = timesteps / dataset_timesteps;
            if (encoding_window <= 0) {
                return;
            }

            const uint base_idx = neuron_id * timesteps;

            for (int column_t = 0; column_t < dataset_timesteps; column_t++) {
                const double encoding_start = column_t * encoding_window;
                const double encoding_end   = encoding_start + encoding_window;

                const double val =
                    data[(observation_idx * num_input_neurons *
                          dataset_timesteps) +
                         (neuron_id * dataset_timesteps) + (column_t)];

                if (val <= 0.0) {
                    continue;
                }

                for (double i = encoding_start; i < encoding_end; i += val) {
                    x[base_idx + (uint)i] = spike_value_factor;
                }
            }
        }

        kernel void risp_forward_kernel(
            global const short* x, global const short* v_thresh,
            global const short* weights, global const uint* delays,
            global const uint* incoming, global const uint* incoming_ids,
            global const uchar* is_input_neuron, global int* v,
            global char* s, global int* v_pre, short v_decay, int v_rest,
            uint num_neurons, uint num_steps, uint timestep,
            uint max_incoming) {
            const uint neuron_id = get_global_id(0);
            if (neuron_id >= num_neurons) {
                return;
            }

            const short V_thresh = v_thresh[neuron_id];
            const int V        = timestep == 0u ? 0 : v[neuron_id];
            const uint idx       = neuron_id * num_steps + timestep;
            const uint inc_count = incoming[neuron_id];

            bool has_event    = false;
            int total_input = 0;

            if (is_input_neuron[neuron_id]) {
                const short input_spike = x[idx];
                if (input_spike != 0) {
                    has_event = true;
                }
                total_input += input_spike;
            }

            // Hoist base indices out of loop
            const uint wts_base = neuron_id * max_incoming;
            const uint del_base = wts_base;
            const uint ids_base = wts_base;

            // Check presynaptic neurons for spikes
            for (uint i = 0u; i < inc_count; i++) {
                const uint inc_off     = ids_base + i;
                const uint incoming_id = incoming_ids[inc_off];
                const short source_ts =
                    (short)timestep - (short)delays[inc_off];

                if (source_ts < 0) {
                    continue;
                }

                const char source_spike =
                    s[incoming_id * num_steps + (uint)source_ts];
                if (source_spike) {
                    has_event = true;
                    total_input += weights[inc_off];
                }
            }

            // Match CPU/event-driven behavior: leak/min-potential only apply
            // when an event arrives for this neuron.
            int V_post = V;
            if (has_event) {
                if (v_decay > 0) {
                    V_post = 0;
                }

                if (V_post < v_rest) {
                    V_post = v_rest;
                }

                V_post += total_input;
            }

            v_pre[idx] = V_post;

            const bool has_spiked = has_event && (V_post >= V_thresh);

            // Reset charge when we spike
            const int V_final = has_spiked ? 0 : V_post;

            v[neuron_id] = V_final;
            s[idx]       = (char)has_spiked;
        }

        kernel void risp_loss_kernel(global const char* s, global float* dL_ds,
                                     global float* correct, global float* loss,
                                     uint num_neurons, uint num_output_neurons,
                                     uint num_steps, uint target_idx) {
            const uint neuron_id = get_global_id(0);
            // Serial for now
            if (neuron_id > 0) {
                return;
            }

            const uint out_base = num_neurons - num_output_neurons;
            float sum           = 0.0f;
            uint max_idx        = 0;
            float max_val       = 0.0f;

            // Pass 1: accumulate spikes, compute mean, find max
            for (uint i = 0; i < num_output_neurons; i++) {
                const uint base_idx = (i + out_base) * num_steps;
                float spike_sum     = 0.0f;
                for (uint t = 0; t < num_steps; t++) {
                    spike_sum += s[base_idx + t];
                }
                dL_ds[i] = spike_sum / (float)num_steps;
                if (dL_ds[i] > max_val) {
                    max_val = dL_ds[i];
                    max_idx = i;
                }
            }

            // Pass 2: softmax
            for (uint i = 0; i < num_output_neurons; i++) {
                dL_ds[i] = exp(dL_ds[i] - max_val);
                sum += dL_ds[i];
            }

            const float inv_sum = 1.0f / sum;
            for (uint i = 0; i < num_output_neurons; i++) {
                dL_ds[i] *= inv_sum;
            }

            *correct += max_idx == target_idx;
            *loss -= log(dL_ds[target_idx] + 1e-8f);

            // Pass 3: dL_ds gradient
            for (uint i = 0; i < num_output_neurons; i++) {
                dL_ds[i] -= (i == target_idx ? 1.0f : 0.0f);
            }
        }

        // Kernel 1: per-neuron gradient computation.
        // Computes neuron_grad and updates future_mem_grad.
        kernel void risp_backward_grad_kernel(
            global const float* dL_ds, global const char* s,
            global const int* v_pre, global const short* v_thresh,
            global const float* gradient_accumulators,
            global const uint* outgoing, global const uchar* is_output_neuron,
            global float* spike_grad_history, global float* future_mem_grad,
            global float* neuron_grad, short v_decay, float v_rest,
            float tau_rho, float scale_rho, uint num_neurons,
            uint num_output_neurons, short num_steps, float scale_factor,
            uint max_outgoing, short timestep) {
            const uint neuron_id = get_global_id(0);
            if (neuron_id >= num_neurons) {
                return;
            }

            const uint idx       = neuron_id * (uint)num_steps + (uint)timestep;
            const uint out_start = num_neurons - num_output_neurons;

            // Sum spike_grad_history for this neuron at this timestep.
            // float spike_grad_sum = spike_grad_history[idx];
            float spike_grad_sum    = 0.0f;
            const uint num_outgoing = outgoing[neuron_id];
            for (int i = 0; i < num_outgoing; i++) {
                spike_grad_sum +=
                    gradient_accumulators[(timestep * num_neurons *
                                           max_outgoing) +
                                          (neuron_id * max_outgoing) + (i)];
            }
            if (is_output_neuron[neuron_id]) {
                spike_grad_sum +=
                    dL_ds[neuron_id - out_start] / (float)num_steps;
            }

            const float dL_dV          = future_mem_grad[neuron_id];
            const float v_pre_t        = v_pre[idx] * scale_factor;
            const float v_thresh_t     = v_thresh[neuron_id] * scale_factor;
            const float dV_post_dV_pre = 1.0f - (s[idx] > 0 ? 1.0f : 0.0f);
            const float dV_post_ds_t =
                v_rest > 0 ? (v_rest - v_pre_t) : -v_pre_t;
            const float ds_t_dV_pre =
                (scale_rho / (2.0f * tau_rho)) *
                exp(-fabs(v_pre_t - v_thresh_t) / tau_rho);

            // Factor ds_t_dV_pre out of common terms
            const float grad =
                dL_dV * dV_post_dV_pre +
                (dL_dV * dV_post_ds_t + spike_grad_sum) * ds_t_dV_pre;

            neuron_grad[neuron_id] = grad;

            if (timestep > 0) {
                const float dV_leak =
                    v_pre_t >= v_rest ? (1.0f - (float)v_decay) : 0.0f;
                if (dV_leak == 0.0f) {
                    future_mem_grad[neuron_id] = 0.0f;
                } else {
                    future_mem_grad[neuron_id] = dV_leak * grad;
                }
            }
        }

        // Kernel 2: per-synapse weight gradient accumulation.
        // Uses neuron_grad to update delta_W and spike_grad_history.
        kernel void risp_backward_delta_w_kernel(
            global const float* neuron_grad, global const char* s,
            global const short* weights, global const uint* delays,
            global const uint* incoming, global const uint* incoming_ids,
            global const uint* gradient_slot, global float* spike_grad_history,
            global float* delta_W, global float* gradient_accumulators,
            uint num_neurons, uint max_incoming, uint max_outgoing,
            short num_steps, float scale_factor, short timestep) {
            const uint global_id = get_global_id(0);

            const uint neuron_id  = global_id / max_incoming;
            const uint synapse_id = global_id % max_incoming;
            if (neuron_id >= num_neurons || synapse_id >= incoming[neuron_id]) {
                return;
            }

            const uint incoming_id = incoming_ids[global_id];
            const short source_ts  = timestep - (short)delays[global_id];
            if (source_ts < 0) {
                return;
            }

            const float grad = neuron_grad[neuron_id];
            const uint src_idx =
                incoming_id * (uint)num_steps + (uint)source_ts;
            const char source_spike = s[src_idx];

            delta_W[global_id] += source_spike * grad;
            gradient_accumulators[(source_ts * num_neurons * max_outgoing) +
                                  (incoming_id * max_outgoing) +
                                  (gradient_slot[(neuron_id * max_incoming) +
                                                 synapse_id])] =
                grad * weights[global_id] * scale_factor;
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
            const float delta  = delta_W[global_id] * inv_batch;
            delta_W[global_id] = 0.0f;

            const float one_minus_beta1 = 1.0f - beta1;
            const float one_minus_beta2 = 1.0f - beta2;
            const float one_minus_b1_t  = 1.0f - b1_t;
            const float one_minus_b2_t  = 1.0f - b2_t;

            // Adam moment updates
            const float new_m =
                beta1 * m_weights[global_id] + one_minus_beta1 * delta;
            const float new_v =
                beta2 * v_weights[global_id] + one_minus_beta2 * delta * delta;
            m_weights[global_id] = new_m;
            v_weights[global_id] = new_v;

            // Biased-corrected moments
            const float mW_hat = new_m / one_minus_b1_t;
            const float vW_hat = sqrt(new_v / one_minus_b2_t + 1.0e-8f);

            // Effective learning rate (warmup on epoch 0)
            const float lr = (epoch == 0)
                                 ? ((batch_start + current_batch_size) /
                                    (float)num_observations) *
                                       learning_rate
                                 : learning_rate;

            float weight = weights[global_id] * scale_factor;
            weight -= lr * mW_hat / vW_hat;
            weight -= lr * decay_rate * weight;
            weight = clamp(weight, -1.0f, 1.0f);
            weight = round(weight / scale_factor);

            weights[global_id] =
                (short)clamp(weight, (float)min_weight, (float)max_weight);
        }

    );
} // ####### End of OpenCL C coe
