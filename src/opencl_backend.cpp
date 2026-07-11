#include "opencl_backend.h"
#include "forward_backward.h"
#include "network_setup.h"
#include "network_utils.h"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace neuro;

/* ------------------------------------------------------------------ */
/* OpenCL timing helpers (GPU profiling via clGetEventProfilingInfo)    */
/* ------------------------------------------------------------------ */
struct KernelTiming {
    string name;
    double total_us;
    double min_us;
    double max_us;
    uint64_t calls;
    KernelTiming() : name(""), total_us(0.0), min_us(1e18), max_us(0.0), calls(0) {}
    KernelTiming(const string& n, double u, uint64_t c)
        : name(n), total_us(u), min_us(u), max_us(u), calls(c) {}
};

static vector<KernelTiming> g_kernels;
static bool g_timing_enabled = false;

static void timing_start(bool enabled) {
    g_timing_enabled = enabled;
    if (!enabled) return;
    g_kernels.clear();
    g_kernels.reserve(8);
}

static void timing_add(const char* name, double us) {
    if (!g_timing_enabled) return;
    for (auto& kt : g_kernels) {
        if (kt.name == name) {
            kt.total_us += us;
            kt.min_us = std::min(kt.min_us, us);
            kt.max_us = std::max(kt.max_us, us);
            kt.calls++;
            return;
        }
    }
    g_kernels.push_back(KernelTiming(name, us, 1));
}

static void timing_print(double total_us) {
    if (!g_timing_enabled || g_kernels.empty()) return;

    printf("\n===== OpenCL Kernel Timing Report (GPU profiling) ===========\n");
    printf("%-28s %10s %10s %10s %8s %10s\n", "Kernel", "Avg(us)", "Min(us)",
           "Max(us)", "Calls", "Share");
    printf("%-28s %10s %10s %10s %8s %10s\n", "----------------------------",
           "----------", "----------", "----------", "--------", "----------");

    for (auto& kt : g_kernels) {
        double avg = (kt.calls > 0) ? (kt.total_us / kt.calls) : 0.0;
        double pct = (total_us > 0) ? (kt.total_us / total_us * 100.0) : 0.0;
        printf("%-28s %10.1f %10.1f %10.1f %8zu %9.2f%%\n", kt.name.c_str(),
               avg, kt.min_us, kt.max_us, kt.calls, pct);
    }
    printf("%-28s %13.3f ms\n", "TOTAL", total_us / 1000.0);
    printf("=============================================================\n\n");
}

static void timed_run(Kernel& kernel, const char* name) {
    if (g_timing_enabled) {
        Event evt;
        kernel.enqueue_run_profiled(&evt);
        evt.wait();
        double us = get_kernel_duration_us(evt);
        timing_add(name, us);
    } else {
        kernel.run();
    }
}

static void encode(Memory<double>& data, const Dataset& d) {
    for (int row = 0; row < d.observations; row++) {
        for (int col = 0; col < d.cols; col++) {
            double x = (d.data[row * d.cols + col] - d.min_vals[col]) /
                       (d.max_vals[col] - d.min_vals[col]);
            double inv_x = 1.0 - x;

            if (x > 0.0) {
                data[(row * d.cols * 2) + (col * 2)] = double(1.0 / x);
            }
            if (inv_x > 0.0) {
                data[(row * d.cols * 2) + (col * 2 + 1)] = double(1.0 / inv_x);
            }
        }
    }
}

static void read_gpu_weights(
    const Memory<short>& gpu_weights, size_t total_neurons, size_t max_incoming,
    const std::vector<std::vector<double>>& state_weights, double scale_factor,
    std::vector<std::vector<double>>& weights) {
    weights.resize(total_neurons);
    for (size_t i = 0; i < total_neurons; i++) {
        size_t inc = state_weights[i].size();
        weights[i].resize(inc);
        for (size_t j = 0; j < inc; j++) {
            weights[i][j] =
                (double)gpu_weights[(i * max_incoming) + j] * scale_factor;
        }
    }
}

static void write_weights_to_network(neuro::Network* n, size_t total_neurons,
                                     bool discrete, double scale_factor,
                                     const std::vector<std::vector<double>>& weights) {
    for (size_t i = 0; i < total_neurons; i++) {
        auto& incoming = n->get_node(i)->incoming;
        for (size_t j = 0; j < incoming.size() && j < weights[i].size(); j++) {
            incoming[j]->set("Weight",
                             weights[i][j] / (discrete ? scale_factor : 1.0));
        }
    }
}

static std::pair<double, double>
cpu_eval_test(neuro::Network* n, const NetworkConfiguration& nc,
              const Dataset& test,
              const std::vector<std::vector<double>>& weights,
              const std::vector<std::vector<int>>& delays,
              const std::vector<double>& thresholds, double rho, double tau) {
    if (test.observations == 0) return {0.0, 0.0};

    TrainingBundle tb(nc.total_neurons, nc.timesteps, nc.output_neurons, rho,
                      tau, &weights, &delays, &thresholds);
    neuro::Processor* p = nullptr;
    load_network(&p, n);

    double total_loss = 0.0;
    size_t total_correct = 0;

    for (int obs = 0; obs < test.observations; obs++) {
        EvaluationResults er = forward(&tb, p, &test, (size_t)obs, &nc);
        total_loss += er.loss;
        total_correct += (size_t)er.correct;
    }

    delete p;
    return {total_loss / test.observations,
            (double)total_correct / test.observations};
}

/* ------------------------------------------------------------------ */
/* OpenclBackend implementation                                         */
/* ------------------------------------------------------------------ */
OpenclBackend::OpenclBackend(const CliConfig& cfg, neuro::Network* n,
                             NetworkConfiguration& nc, const Dataset& train,
                             const Dataset& test, TrainingState* state,
                             size_t max_incoming, size_t max_outgoing,
                             size_t batch_size, double learning_rate,
                             double decay_rate, double rho, double tau)
    : m_cfg(cfg), m_n(n), m_nc(nc), m_train(train), m_test(test),
      m_state(state), m_max_incoming(max_incoming), m_max_outgoing(max_outgoing),
      m_batch_size(batch_size), m_learning_rate(learning_rate),
      m_decay_rate(decay_rate), m_rho(rho), m_tau(tau),
      m_b1_t(1.0), m_b2_t(1.0) {

    Device device(select_device_with_most_flops());
    const size_t encode_work_size = nc.input_neurons;
    const size_t forward_work_size = nc.total_neurons;
    const size_t loss_work_size = nc.output_neurons;
    const size_t backward_grad_work_size = nc.total_neurons;
    const size_t backward_delta_w_work_size = nc.total_neurons * nc.max_incoming;
    const size_t weight_updates_work_size = nc.total_neurons * nc.max_incoming;

    m_x.reset(new Memory<short>(device, nc.input_neurons * nc.timesteps));
    m_data.reset(new Memory<double>(device, train.observations * train.cols * 2));
    m_test_data.reset(new Memory<double>(device, test.observations * test.cols * 2));
    m_v_thresh.reset(new Memory<short>(device, nc.total_neurons));
    m_weights.reset(new Memory<short>(device, nc.total_neurons * nc.max_incoming));
    m_delays.reset(new Memory<uint>(device, nc.total_neurons * nc.max_incoming));
    m_incoming.reset(new Memory<uint>(device, nc.total_neurons));
    m_incoming_ids.reset(new Memory<uint>(device, nc.total_neurons * nc.max_incoming));
    m_is_input_neuron.reset(new Memory<uchar>(device, nc.total_neurons));
    m_is_output_neuron.reset(new Memory<uchar>(device, nc.total_neurons));
    m_v.reset(new Memory<short>(device, nc.total_neurons));
    m_s.reset(new Memory<char>(device, nc.total_neurons * nc.timesteps));
    m_v_pre.reset(new Memory<short>(device, nc.total_neurons * nc.timesteps));
    m_dL_ds.reset(new Memory<float>(device, nc.output_neurons));
    m_correct.reset(new Memory<float>(device, 1));
    m_loss.reset(new Memory<float>(device, 1));
    m_spike_grad_history.reset(new Memory<float>(device, nc.total_neurons * nc.timesteps));
    m_future_mem_grad.reset(new Memory<float>(device, nc.total_neurons));
    m_delta_W.reset(new Memory<float>(device, nc.total_neurons * nc.max_incoming));
    m_neuron_grad.reset(new Memory<float>(device, nc.total_neurons));
    m_m_weights.reset(new Memory<float>(device, nc.total_neurons * nc.max_incoming));
    m_v_weights.reset(new Memory<float>(device, nc.total_neurons * nc.max_incoming));
    m_outgoing.reset(new Memory<uint>(device, nc.total_neurons));
    m_gradient_slot.reset(new Memory<uint>(device, nc.total_neurons * nc.max_incoming));
    m_gradient_accumulators.reset(
        new Memory<float>(device, nc.timesteps * nc.total_neurons * nc.max_outgoing));

    m_encode_kernel.reset(new Kernel(
        device, encode_work_size, "risp_encode_inputs_kernel",
        *m_x, *m_data, (int)train.cols, (int)nc.input_neurons,
        (int)nc.timesteps, (uint)0, (short)nc.spike_value_factor));

    m_forward_kernel.reset(new Kernel(
        device, forward_work_size, "risp_forward_kernel", *m_x, *m_v_thresh,
        *m_weights, *m_delays, *m_incoming, *m_incoming_ids,
        *m_is_input_neuron, *m_v, *m_s, *m_v_pre, (short)nc.leak,
        (short)(nc.min_potential / nc.scale_factor), (uint)nc.total_neurons,
        (uint)nc.timesteps, (uint)0, (uint)nc.max_incoming));

    m_loss_kernel.reset(new Kernel(
        device, loss_work_size, "risp_loss_kernel", *m_s, *m_dL_ds,
        *m_correct, *m_loss, (uint)nc.total_neurons, (uint)nc.output_neurons,
        (uint)nc.timesteps, (uint)0));

    m_backward_grad_kernel.reset(new Kernel(
        device, backward_grad_work_size, "risp_backward_grad_kernel",
        *m_dL_ds, *m_s, *m_v_pre, *m_v_thresh, *m_gradient_accumulators,
        *m_outgoing, *m_is_output_neuron, *m_spike_grad_history,
        *m_future_mem_grad, *m_neuron_grad, (short)nc.leak,
        (float)nc.min_potential, (float)tau, (float)rho, (uint)nc.total_neurons,
        (uint)nc.output_neurons, (short)nc.timesteps, (float)nc.scale_factor,
        (uint)nc.max_outgoing, (short)0));

    m_backward_delta_w_kernel.reset(new Kernel(
        device, backward_delta_w_work_size, "risp_backward_delta_w_kernel",
        *m_neuron_grad, *m_s, *m_weights, *m_delays, *m_incoming,
        *m_incoming_ids, *m_gradient_slot, *m_spike_grad_history, *m_delta_W,
        *m_gradient_accumulators, (uint)nc.total_neurons, (uint)nc.max_incoming,
        (uint)nc.max_outgoing, (short)nc.timesteps, (float)nc.scale_factor,
        (short)0));

    m_weight_updates_kernel.reset(new Kernel(
        device, weight_updates_work_size, "weight_updates_kernel", *m_incoming,
        *m_m_weights, *m_v_weights, *m_delta_W, *m_weights,
        (uint)nc.total_neurons, (uint)nc.max_incoming, (float)learning_rate,
        (float)decay_rate, (uint)1, (uint)batch_size, (uint)0, (uint)0,
        (float)0.9f, (float)0.999f, (float)0.0f, (float)0.0f,
        (uint)nc.timesteps, (uint)train.observations, (float)nc.scale_factor,
        (short)nc.min_weight, (short)nc.max_weight, (int)nc.steps));

    // Encode data
    encode(*m_data, train);
    encode(*m_test_data, test);

    // Initialize GPU buffers from host state
    for (size_t i = 0; i < nc.total_neurons; i++) {
        neuro::Node* node = n->get_node(i);

        m_v_thresh->operator[](i) = state->thresholds[i] / nc.scale_factor;
        m_incoming->operator[](i) = state->weights[i].size();
        m_is_input_neuron->operator[](i) = i < nc.input_neurons;
        m_is_output_neuron->operator[](i) = i >= nc.input_neurons + nc.hidden_neurons;

        for (size_t j = 0; j < state->weights[i].size(); j++) {
            (*m_weights)[(i * max_incoming) + j] =
                state->weights[i][j] / nc.scale_factor;
            (*m_delays)[(i * max_incoming) + j] = state->delays[i][j];
            (*m_incoming_ids)[(i * max_incoming) + j] =
                n->get_node(i)->incoming[j]->from->id;
        }

        for (size_t j = 0; j < node->incoming.size(); j++) {
            neuro::Edge* edge = node->incoming[j];
            size_t incoming_id = edge->from->id;

            (*m_gradient_slot)[i * nc.max_incoming + j] = (*m_outgoing)[incoming_id];
            (*m_outgoing)[incoming_id]++;
        }
    }

    m_m_weights->reset();
    m_v_weights->reset();

    m_data->write_to_device();
    m_test_data->write_to_device();
    m_v_thresh->write_to_device();
    m_weights->write_to_device();
    m_delays->write_to_device();
    m_incoming->write_to_device();
    m_incoming_ids->write_to_device();
    m_is_input_neuron->write_to_device();
    m_is_output_neuron->write_to_device();
    m_m_weights->write_to_device();
    m_v_weights->write_to_device();
    m_outgoing->write_to_device();
    m_gradient_slot->write_to_device();

    // Init batch order
    m_batch_order.resize(train.observations);
    for (int i = 0; i < train.observations; i++) {
        m_batch_order[i] = (size_t)i;
    }

    timing_start(cfg.opencl_timings);
    m_t_start = chrono::high_resolution_clock::now();
}

void OpenclBackend::do_one_epoch(size_t epoch) {
    double epoch_loss = 0.0;
    size_t epoch_correct = 0;
    m_correct->reset();
    m_loss->reset();

    // Shuffle batch order each epoch
    for (int i = 0; i < m_train.observations; i++) {
        int j = rand() % m_train.observations;
        size_t tmp = m_batch_order[i];
        m_batch_order[i] = m_batch_order[j];
        m_batch_order[j] = tmp;
    }

    // Mini-batch SGD loop
    for (int batch_start = 0; batch_start < m_train.observations;
         batch_start += (int)m_batch_size) {
        size_t current_batch_size =
            min(m_batch_size, (size_t)(m_train.observations - batch_start));

        // Reset accumulators for this batch
        m_delta_W->reset();

        for (size_t b = 0; b < current_batch_size; b++) {
            size_t obs = m_batch_order[(size_t)batch_start + b];

            m_x->reset();
            m_v->reset();
            m_s->reset();
            m_v_pre->reset();
            m_dL_ds->reset();
            m_spike_grad_history->reset();
            m_future_mem_grad->reset();
            m_gradient_accumulators->reset();

            // Encode data
            m_encode_kernel->set_parameters(5, (uint)obs);
            timed_run(*m_encode_kernel, "encode");

            // Forward pass
            for (size_t t = 0; t < m_nc.timesteps; t++) {
                m_forward_kernel->set_parameters(14, (uint)t);
                timed_run(*m_forward_kernel, "forward");
            }

            // Loss
            m_loss_kernel->set_parameters(7, (uint)(m_train.labels[obs]));
            timed_run(*m_loss_kernel, "loss");

            // Backwards
            for (short t = m_nc.timesteps - 1; t >= 0; t--) {
                m_backward_grad_kernel->set_parameters(19, (short)t);
                timed_run(*m_backward_grad_kernel, "backward_grad");
                m_backward_delta_w_kernel->set_parameters(15, (short)t);
                timed_run(*m_backward_delta_w_kernel, "backward_delta_w");
            }
        }

        // Weight updates
        m_b1_t *= 0.9;
        m_b2_t *= 0.999;
        m_weight_updates_kernel->set_parameters(9, (uint)current_batch_size,
                                                 (uint)m_batch_size);
        m_weight_updates_kernel->set_parameters(11, (uint)batch_start);
        m_weight_updates_kernel->set_parameters(12, (uint)epoch);
        m_weight_updates_kernel->set_parameters(15, (float)m_b1_t, (float)m_b2_t);
        timed_run(*m_weight_updates_kernel, "weight_updates");
    }

    m_correct->read_from_device();
    m_loss->read_from_device();
    epoch_loss += (*m_loss)[0];
    epoch_correct += (size_t)(*m_correct)[0];

    double avg_train_loss = epoch_loss / (double)m_train.observations;
    double avg_train_acc = epoch_correct / (double)m_train.observations;

    if (avg_train_loss < m_stats.best_train_loss) m_stats.best_train_loss = avg_train_loss;
    if (avg_train_acc > m_stats.best_train_acc) m_stats.best_train_acc = avg_train_acc;

    // Test evaluation
    double epoch_test_loss = 0.0;
    size_t epoch_test_correct = 0;

    if (m_test.observations > 0) {
        m_correct->reset();
        m_loss->reset();

        m_encode_kernel->set_parameters(1, *m_test_data);

        for (int obs = 0; obs < (int)m_test.observations; obs++) {
            m_x->reset();
            m_v->reset();
            m_s->reset();
            m_v_pre->reset();
            m_dL_ds->reset();

            m_encode_kernel->set_parameters(2, (int)m_test.cols);
            m_encode_kernel->set_parameters(5, (uint)obs);
            timed_run(*m_encode_kernel, "encode");

            for (size_t t = 0; t < m_nc.timesteps; t++) {
                m_forward_kernel->set_parameters(14, (uint)t);
                timed_run(*m_forward_kernel, "forward");
            }

            m_loss_kernel->set_parameters(7, (uint)(m_test.labels[obs]));
            timed_run(*m_loss_kernel, "loss");
        }

        m_correct->read_from_device();
        m_loss->read_from_device();
        epoch_test_correct += (size_t)(*m_correct)[0];
        epoch_test_loss += (*m_loss)[0];

        m_encode_kernel->set_parameters(1, *m_data);
    }

    double avg_test_loss = m_test.observations > 0
                               ? epoch_test_loss / (double)m_test.observations
                               : 0.0;
    double avg_test_acc = m_test.observations > 0
                              ? epoch_test_correct / (double)m_test.observations
                              : 0.0;

    if (m_test.observations > 0) {
        if (avg_test_loss < m_stats.best_test_loss) m_stats.best_test_loss = avg_test_loss;
        if (avg_test_acc > m_stats.best_test_acc) m_stats.best_test_acc = avg_test_acc;
    }

    m_stats.train_acc = avg_train_acc;
    m_stats.train_loss = avg_train_loss;
    m_stats.test_acc = avg_test_acc;
    m_stats.test_loss = avg_test_loss;

    // Periodic CPU eval
    if (m_cfg.cpu_eval_interval > 0 && (epoch + 1) % m_cfg.cpu_eval_interval == 0) {
        m_weights->read_from_device();
        std::vector<std::vector<double>> cpu_weights;
        read_gpu_weights(*m_weights, m_nc.total_neurons, m_max_incoming,
                         m_state->weights, m_nc.scale_factor, cpu_weights);
        write_weights_to_network(m_n, m_nc.total_neurons, m_nc.discrete,
                                 m_nc.scale_factor, cpu_weights);
        std::pair<double, double> cpu_result =
            cpu_eval_test(m_n, m_nc, m_test, cpu_weights, m_state->delays,
                          m_state->thresholds, m_rho, m_tau);
        printf("  [CPU eval @ epoch %4zu] Test Loss: %10g, Test Acc: %10g\n",
               epoch + 1, cpu_result.first, cpu_result.second);
    }
}

std::pair<double, double> OpenclBackend::run_final_cpu_eval() {
    // Read fresh GPU weights — m_state->weights is not updated during GPU training
    m_weights->read_from_device();
    std::vector<std::vector<double>> cpu_weights;
    read_gpu_weights(*m_weights, m_nc.total_neurons, m_max_incoming,
                     m_state->weights, m_nc.scale_factor, cpu_weights);
    return cpu_eval_test(m_n, m_nc, m_test, cpu_weights, m_state->delays,
                         m_state->thresholds, m_rho, m_tau);
}

OpenclBackend::~OpenclBackend() {
    auto t_end = chrono::high_resolution_clock::now();
    double total_us =
        chrono::duration<double, std::micro>(t_end - m_t_start).count();
    timing_print(total_us);

    // Read GPU weights back to host for state
    m_weights->read_from_device();
    read_gpu_weights(*m_weights, m_nc.total_neurons, m_max_incoming,
                     m_state->weights, m_nc.scale_factor, m_state->weights);
}

TrainingStats OpenclBackend::get_stats() const { return m_stats; }

void OpenclBackend::update_weights(neuro::Network* network) {
    // Read GPU weights back to host — all weight mutation happens on GPU
    m_weights->read_from_device();
    read_gpu_weights(*m_weights, m_nc.total_neurons, m_max_incoming,
                     m_state->weights, m_nc.scale_factor, m_state->weights);
    write_weights_to_network(network, m_nc.total_neurons, m_nc.discrete,
                             m_nc.scale_factor, m_state->weights);
}
