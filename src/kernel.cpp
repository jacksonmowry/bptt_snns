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

            ushort base_idx = neuron_id * timesteps;
            for (double i = 0.0; i < (double)timesteps; i += val) {
                x[base_idx + (ushort)i] = spike_value_factor;
            }
        }

        kernel void risp_forward_kernel(
            global const short* x, global const short* v_thresh,
            global const short* weights, global const ushort* delays,
            global const ushort* incoming, global const ushort* incoming_ids,
            global const uchar* is_input_neuron, global short* v,
            global char* s, global short* v_pre, short v_decay, short v_rest,
            ushort num_neurons, ushort num_steps, ushort timestep,
            ushort max_incoming) {
            const uint neuron_id = get_global_id(0);
            if (neuron_id >= (uint)num_neurons) {
                return;
            }

            short V_thresh = v_thresh[neuron_id];
            short V_rest   = v_rest;
            short V        = timestep == 0 ? 0 : v[neuron_id];
            ushort idx     = neuron_id * num_steps + timestep;

            // Binary leak
            if (v_decay > 0) {
                V = 0;
            }

            if (V < V_rest) {
                V = V_rest;
            }

            if (is_input_neuron[neuron_id]) {
                V += x[idx];
            }

            // Check presynaptic neurons for spikes
            for (int i = 0; i < incoming[neuron_id]; i++) {
                const ushort incoming_id =
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
                V += source_spike ? weight : 0;
            }

            v_pre[idx] = V;

            const bool has_spiked = V >= V_thresh;

            // Reset charge when we spike
            if (has_spiked) {
                V = 0;
            }

            // Final min potential reset
            if (V < V_rest) {
                V = V_rest;
            }

            v[neuron_id] = V;
            s[idx]       = (char)has_spiked;
        }

        kernel void risp_loss_kernel(
            global const char* s, global float* dL_ds, global double* correct,
            global double* loss, ushort num_neurons, ushort num_output_neurons,
            ushort num_steps, ushort target_idx) {
            const uint neuron_id = get_global_id(0);
            // Serial for now
            if (neuron_id > 0) {
                return;
            }

            float sum     = 0.0;
            int max_idx    = 0;
            float max_val = 0;

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
            *loss -= log((double)dL_ds[target_idx] + 1e-8);

            for (int i = 0; i < num_output_neurons; i++) {
                dL_ds[i] -= (i == target_idx ? 1.0 : 0.0);
            }
        }

        kernel void risp_backward_kernel(
            global const float* dL_ds, global const char* s,
            global const short* v_pre, global const short* v_thresh,
            global const uchar* is_output_neuron, global const short* weights,
            global const ushort* delays, global const ushort* incoming,
            global const ushort* incoming_ids,
            global float* spike_grad_history,
            global double* voltage_grad_history, global double* future_mem_grad,
            global double* delta_W,
            short v_decay, short v_rest, double tau_rho,
            double scale_rho, ushort num_neurons, ushort num_output_neurons,
            short num_steps, ushort max_incoming, double scale_factor,
            short timestep) {
            const uint global_id = get_global_id(0);

            const uint neuron_id  = global_id / max_incoming;
            const uint synapse_id = global_id % max_incoming;
            if (neuron_id >= num_neurons || synapse_id >= incoming[neuron_id]) {
                return;
            }

            const int base                    = neuron_id * num_steps;
            const int idx = base + timestep;
            const bool has_decay        = v_decay > 0;
            const double tau_rho_scaled = tau_rho;


            // Sum spike_grad_history over all synapses of this neuron at this timestep
            // Layout: [t * num_neurons * max_incoming + n * max_incoming + s]
            double spike_grad_sum = spike_grad_history[idx];
            if (is_output_neuron[neuron_id]) {
                // Output neuron: spread dL_ds across all incoming synapse slots
                spike_grad_sum +=
                    dL_ds[neuron_id - (num_neurons - num_output_neurons)] /
                    (double)num_steps;
            }

            const double dL_dV =
                voltage_grad_history[idx] + future_mem_grad[neuron_id];
            const double v_pre_t        = v_pre[idx] * scale_factor;
            const double dy_t_dV_post   = 1.0;
            const double dV_post_dV_pre = 1.0 - (s[idx] > 0 ? 1.0 : 0.0);
            const double dV_pre_dx_t    = 1.0;
            const double dV_post_ds_t =
                ((v_rest * scale_factor) > 0)
                ? ((v_rest * scale_factor) - v_pre_t)
                : (-v_pre_t);
            const double ds_t_dV_pre =
                ((scale_rho / (2.0 * tau_rho_scaled)) *
                 exp(-fabs(v_pre_t - (v_thresh[neuron_id] * scale_factor)) /
                     tau_rho_scaled));
            const double grad =
                (dL_dV * dV_post_dV_pre * dV_pre_dx_t) +
                (dL_dV * dV_post_ds_t * ds_t_dV_pre * dV_pre_dx_t) +
                (spike_grad_sum * ds_t_dV_pre * dV_pre_dx_t);

            if (timestep > 0) {
                const double dV_pre_dV_leak = 1.0;
                const double dV_leak_dV_t1 =
                    (v_pre_t >= (v_rest * scale_factor)) ? (1.0 - v_decay)
                    : 0.0;

                future_mem_grad[neuron_id] +=
                    (dL_dV * dV_post_dV_pre * dV_pre_dV_leak * dV_leak_dV_t1) +
                    (dL_dV * dV_post_ds_t * ds_t_dV_pre * dV_pre_dV_leak *
                     dV_leak_dV_t1) +
                    (spike_grad_sum * ds_t_dV_pre * dV_pre_dV_leak *
                     dV_leak_dV_t1);
            }

            ushort incoming_id = incoming_ids[global_id];
            short source_ts   = timestep - delays[global_id];
            double weight      = weights[global_id] * scale_factor;

            if (source_ts < 0) {
                return;
            }

            char source_spike = s[incoming_id * num_steps + source_ts];

            delta_W[global_id] += source_spike * grad;
            // 3D layout - each thread writes to unique [source_ts][incoming_id][synapse_id] slot
            // No race: synapse_id is unique per thread (global_id = neuron_id * max_incoming + synapse_id)
            atomic_fetch_add((_Atomic float*)&spike_grad_history[incoming_id * num_steps + source_ts], (float)(grad*weight));
        }

        kernel void weight_updates_kernel(
            global const ushort* incoming, global double* m_weights,
            global double* v_weights, global double* delta_W,
            global short* weights, ushort num_neurons, ushort max_incoming,
            double learning_rate, double decay_rate, ushort current_batch_size,
            ushort batch_size, uint batch_start, uint epoch, double beta1,
            double beta2, double b1_t, double b2_t, ushort timesteps,
            uint num_observations, double scale_factor, short min_weight,
            short max_weight, int steps) {
            const uint global_id  = get_global_id(0);
            const uint neuron_id  = global_id / max_incoming;
            const uint synapse_id = global_id % max_incoming;
            if (global_id >= num_neurons * max_incoming ||
                neuron_id >= num_neurons || synapse_id >= incoming[neuron_id]) {
                return;
            }

            const double inv_batch =
                1.0 / ((double)current_batch_size * (double)timesteps);
            const double delta = delta_W[global_id] * inv_batch;
            delta_W[global_id] = 0.0;

            const double new_m_weight = m_weights[global_id] =
                beta1 * m_weights[global_id] + (1.0 - beta1) * delta;
            const double new_v_weight = v_weights[global_id] =
                beta2 * v_weights[global_id] + (1.0 - beta2) * (delta * delta);

            const double mW_hat = new_m_weight / (1.0 - b1_t);
            const double vW_hat = new_v_weight / (1.0 - b2_t);

            double lr;
            if (false && epoch == 0) {
                lr = ((batch_start + current_batch_size) /
                      (double)num_observations) *
                    learning_rate;
            } else {
                lr = learning_rate;
            }

            double weight = weights[global_id] * scale_factor;
            weight -= lr * mW_hat / sqrt(vW_hat + 1.0e-8);
            weight -= lr * decay_rate * weight;
            weight = clamp(weight, -1.0, 1.0);
            weight = (round(weight / (2.0 / (double)steps))) * (2.0 / (double)steps);

            weights[global_id] =
                clamp((short)(weight / scale_factor), min_weight, max_weight);
        }

    );
} // ####### End of OpenCL C coe
