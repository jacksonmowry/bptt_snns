#include "opencl_training.h"
#include "forward_backward.h"
#include "network_setup.h"
#include "network_utils.h"
#include "opencl.hpp"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace neuro;

/* ------------------------------------------------------------------ */
/* OpenCL timing helpers                                                */
/* ------------------------------------------------------------------ */
struct KernelTiming {
    string name;
    double total_us; // accumulated microseconds
    uint64_t calls;  // number of .run() invocations
    KernelTiming() : name(""), total_us(0.0), calls(0) {}
    KernelTiming(const string& n, double u, uint64_t c)
        : name(n), total_us(u), calls(c) {}
};

static vector<KernelTiming> g_kernels;
static bool g_timing_enabled = false;

static void timing_start(bool enabled) {
    g_timing_enabled = enabled;
    if (!enabled) {
        return;
    }
    g_kernels.clear();
    g_kernels.reserve(8);
}

static void timing_add(const char* name, double us) {
    if (!g_timing_enabled) {
        return;
    }
    for (auto& kt : g_kernels) {
        if (kt.name == name) {
            kt.total_us += us;
            kt.calls++;
            return;
        }
    }
    g_kernels.push_back(KernelTiming(name, us, 1));
}

static void timing_print(double total_us) {
    if (!g_timing_enabled || g_kernels.empty()) {
        return;
    }

    printf(
        "\n===== OpenCL Kernel Timing Report ============================\n");
    printf("%-30s %14s %10s %10s\n", "Kernel", "Time (ms)", "Calls", "Share");
    printf("%-30s %14s %10s %10s\n", "------------------------------",
           "--------------", "----------", "----------");

    for (auto& kt : g_kernels) {
        double pct = (total_us > 0) ? (kt.total_us / total_us * 100.0) : 0.0;
        printf("%-30s %13.3f ms %10zu %9.2f%%\n", kt.name.c_str(),
               kt.total_us / 1000.0, kt.calls, pct);
    }
    printf("%-30s %13.3f ms\n", "TOTAL", total_us / 1000.0);
    printf("=============================================================\n\n");
}

template <typename... Args>
static void timed_run(Kernel& kernel, const char* name, Args&&... args) {
    auto t0 = chrono::high_resolution_clock::now();
    kernel.run(std::forward<Args>(args)...);
    auto t1   = chrono::high_resolution_clock::now();
    double us = chrono::duration<double, std::micro>(t1 - t0).count();
    timing_add(name, us);
}

static void encode(Memory<double>& data, const Dataset& d) {
    for (int row = 0; row < d.observations; row++) {
        for (int col = 0; col < d.cols; col++) {
            double x     = (d.data[row * d.cols + col] - d.min_vals[col]) /
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

/* Read GPU weights back into state->weights (double, scale_factor-applied). */
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

/* Write weights from a vector-of-vectors back into Network edges. */
static void
write_weights_to_network(neuro::Network* n, size_t total_neurons, bool discrete,
                         double scale_factor,
                         const std::vector<std::vector<double>>& weights) {
    for (size_t i = 0; i < total_neurons; i++) {
        auto& incoming = n->get_node(i)->incoming;
        for (size_t j = 0; j < incoming.size() && j < weights[i].size(); j++) {
            incoming[j]->set("Weight",
                             weights[i][j] / (discrete ? scale_factor : 1.0));
        }
    }
}

/* Run CPU-only forward+loss on test set. Returns {loss, accuracy}. */
static std::pair<double, double>
cpu_eval_test(neuro::Network* n, const NetworkConfiguration& nc,
              const Dataset& test,
              const std::vector<std::vector<double>>& weights,
              const std::vector<std::vector<int>>& delays,
              const std::vector<double>& thresholds, double rho, double tau) {
    if (test.observations == 0) {
        return {0.0, 0.0};
    }

    TrainingBundle tb(nc.total_neurons, nc.timesteps, nc.output_neurons, rho,
                      tau, &weights, &delays, &thresholds);

    neuro::Processor* p = nullptr;
    load_network(&p, n);

    double total_loss    = 0.0;
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

/* Export network to JSON file. */
static void export_network_json(neuro::Network* n, const std::string& path) {
    nlohmann::json j;
    n->to_json(j);
    std::ofstream fout(path);
    if (!fout) {
        fprintf(stderr, "Failed to open %s for writing\n", path.c_str());
        exit(1);
    }
    fout << j.dump(2) << std::endl;
    fout.close();
}

bool opencl_train(const CliConfig& cfg, neuro::Network* n,
                  const NetworkConfiguration& nc, const Dataset& train,
                  const Dataset& test, TrainingState* state,
                  size_t max_incoming, size_t max_outgoing, size_t epochs,
                  size_t batch_size, double learning_rate, double decay_rate,
                  double rho, double tau, bool export_json) {

    Device device(select_device_with_most_flops());
    const size_t encode_work_size        = nc.input_neurons;
    const size_t forward_work_size       = nc.total_neurons;
    const size_t loss_work_size          = nc.output_neurons;
    const size_t backward_grad_work_size = nc.total_neurons;
    const size_t backward_delta_w_work_size =
        nc.total_neurons * nc.max_incoming;
    const size_t weight_updates_work_size = nc.total_neurons * nc.max_incoming;

    Memory<short> x(device, nc.input_neurons * nc.timesteps);
    Memory<double> data(device, train.observations * train.cols * 2);
    Memory<double> test_data(device, test.observations * test.cols * 2);
    Memory<short> v_thresh(device, nc.total_neurons);
    Memory<short> weights(device, nc.total_neurons * nc.max_incoming);
    Memory<uint> delays(device, nc.total_neurons * nc.max_incoming);
    Memory<uint> incoming(device, nc.total_neurons);
    Memory<uint> incoming_ids(device, nc.total_neurons * nc.max_incoming);
    Memory<uchar> is_input_neuron(device, nc.total_neurons);
    Memory<uchar> is_output_neuron(device, nc.total_neurons);
    Memory<short> v(device, nc.total_neurons);
    Memory<char> s(device, nc.total_neurons * nc.timesteps);
    Memory<short> v_pre(device, nc.total_neurons * nc.timesteps);
    Memory<float> dL_ds(device, nc.output_neurons);
    Memory<float> correct(device, 1);
    Memory<float> loss(device, 1);
    Memory<float> spike_grad_history(device, nc.total_neurons * nc.timesteps);
    Memory<float> future_mem_grad(device, nc.total_neurons);
    Memory<float> delta_W(device, nc.total_neurons * nc.max_incoming);
    Memory<float> neuron_grad(device, nc.total_neurons);
    Memory<float> m_weights(device, nc.total_neurons * nc.max_incoming);
    Memory<float> v_weights(device, nc.total_neurons * nc.max_incoming);
    Memory<uint> outgoing(device, nc.total_neurons);
    Memory<uint> gradient_slot(device, nc.total_neurons * nc.max_incoming);
    Memory<float> gradient_accumulators(
        device, nc.timesteps * nc.total_neurons * nc.max_outgoing);

    Kernel encode_kernel(device, encode_work_size, "risp_encode_inputs_kernel",
                         x, data, (int)train.cols, (int)nc.input_neurons,
                         (int)nc.timesteps, (uint)0,
                         (short)nc.spike_value_factor);

    Kernel forward_kernel(device, forward_work_size, "risp_forward_kernel", x,
                          v_thresh, weights, delays, incoming, incoming_ids,
                          is_input_neuron, v, s, v_pre, (short)nc.leak,
                          (short)(nc.min_potential / nc.scale_factor),
                          (uint)nc.total_neurons, (uint)nc.timesteps, (uint)0,
                          (uint)nc.max_incoming);

    Kernel loss_kernel(device, loss_work_size, "risp_loss_kernel", s, dL_ds,
                       correct, loss, (uint)nc.total_neurons,
                       (uint)nc.output_neurons, (uint)nc.timesteps, (uint)0);

    Kernel backward_grad_kernel(
        device, backward_grad_work_size, "risp_backward_grad_kernel", dL_ds, s,
        v_pre, v_thresh, gradient_accumulators, outgoing, is_output_neuron,
        spike_grad_history, future_mem_grad, neuron_grad, (short)nc.leak,
        (float)nc.min_potential, (float)tau, (float)rho, (uint)nc.total_neurons,
        (uint)nc.output_neurons, (short)nc.timesteps, (float)nc.scale_factor,
        (uint)nc.max_outgoing, (short)0);

    Kernel backward_delta_w_kernel(
        device, backward_delta_w_work_size, "risp_backward_delta_w_kernel",
        neuron_grad, s, weights, delays, incoming, incoming_ids, gradient_slot,
        spike_grad_history, delta_W, gradient_accumulators,
        (uint)nc.total_neurons, (uint)nc.max_incoming, (uint)nc.max_outgoing,
        (short)nc.timesteps, (float)nc.scale_factor, (short)0);

    Kernel weight_updates_kernel(
        device, weight_updates_work_size, "weight_updates_kernel", incoming,
        m_weights, v_weights, delta_W, weights, (uint)nc.total_neurons,
        (uint)nc.max_incoming, (float)learning_rate, (float)decay_rate, (uint)1,
        (uint)batch_size, (uint)0, (uint)0, (float)0.9f, (float)0.999f,
        (float)0.0f, (float)0.0f, (uint)nc.timesteps, (uint)train.observations,
        (float)nc.scale_factor, (short)nc.min_weight, (short)nc.max_weight,
        (int)nc.steps);

    // Temporary handling for numeric instability
    encode(data, train);
    encode(test_data, test);

    for (size_t i = 0; i < nc.total_neurons; i++) {
        neuro::Node* node = n->get_node(i);

        v_thresh[i]         = state->thresholds[i] / nc.scale_factor;
        incoming[i]         = state->weights[i].size();
        is_input_neuron[i]  = i < nc.input_neurons;
        is_output_neuron[i] = i >= nc.input_neurons + nc.hidden_neurons;

        for (size_t j = 0; j < state->weights[i].size(); j++) {
            weights[(i * max_incoming) + j] =
                state->weights[i][j] / nc.scale_factor;
            delays[(i * max_incoming) + j] = state->delays[i][j];
            incoming_ids[(i * max_incoming) + j] =
                n->get_node(i)->incoming[j]->from->id;
        }

        // Outgoing buffers init
        for (size_t j = 0; j < node->incoming.size(); j++) {
            neuro::Edge* edge  = node->incoming[j];
            size_t incoming_id = edge->from->id;

            gradient_slot[i * nc.max_incoming + j] = outgoing[incoming_id];
            outgoing[incoming_id]++;
        }
    }

    m_weights.reset();
    v_weights.reset();

    data.write_to_device();
    test_data.write_to_device();
    v_thresh.write_to_device();
    weights.write_to_device();
    delays.write_to_device();
    incoming.write_to_device();
    incoming_ids.write_to_device();
    is_input_neuron.write_to_device();
    is_output_neuron.write_to_device();
    m_weights.write_to_device();
    v_weights.write_to_device();
    outgoing.write_to_device();
    gradient_slot.write_to_device();

    double b1_t = 1.0;
    double b2_t = 1.0;

    // Shuffle order for SGD
    vector<size_t> batch_order(train.observations);
    for (int i = 0; i < train.observations; i++) {
        batch_order[i] = (size_t)i;
    }

    double best_acc       = 0.0;
    double best_loss      = DBL_MAX;
    double best_test_acc  = 0.0;
    double best_test_loss = DBL_MAX;

    timing_start(cfg.opencl_timings);
    auto t_start = chrono::high_resolution_clock::now();

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss    = 0.0;
        size_t epoch_correct = 0;
        correct.reset();
        loss.reset();

        // Shuffle batch order each epoch
        for (int i = 0; i < train.observations; i++) {
            int j          = rand() % train.observations;
            size_t tmp     = batch_order[i];
            batch_order[i] = batch_order[j];
            batch_order[j] = tmp;
        }

        // Mini-batch SGD loop
        for (int batch_start = 0; batch_start < train.observations;
             batch_start += (int)batch_size) {
            size_t current_batch_size =
                min(batch_size, (size_t)(train.observations - batch_start));

            // Reset accumulators for this batch
            delta_W.reset();

            for (size_t b = 0; b < current_batch_size; b++) {
                size_t obs = batch_order[(size_t)batch_start + b];

                x.reset();
                v.reset();
                s.reset();
                v_pre.reset();
                dL_ds.reset();
                spike_grad_history.reset();
                future_mem_grad.reset();
                gradient_accumulators.reset();

                // Encode data
                encode_kernel.set_parameters(5, (uint)obs);
                timed_run(encode_kernel, "encode");

                // Forward pass
                for (size_t t = 0; t < nc.timesteps; t++) {
                    forward_kernel.set_parameters(14, (uint)t);
                    timed_run(forward_kernel, "forward");
                }

                // Loss
                loss_kernel.set_parameters(7, (uint)(train.labels[obs]));
                timed_run(loss_kernel, "loss");

                // Backwards
                for (short t = nc.timesteps - 1; t >= 0; t--) {
                    backward_grad_kernel.set_parameters(19, (short)t);
                    timed_run(backward_grad_kernel, "backward_grad");
                    backward_delta_w_kernel.set_parameters(15, (short)t);
                    timed_run(backward_delta_w_kernel, "backward_delta_w");
                }
            }

            // Weight updates
            b1_t *= 0.9;
            b2_t *= 0.999;
            weight_updates_kernel.set_parameters(9, (uint)current_batch_size,
                                                 (uint)batch_size);
            weight_updates_kernel.set_parameters(11, (uint)batch_start);
            weight_updates_kernel.set_parameters(12, (uint)epoch);
            weight_updates_kernel.set_parameters(15, (float)b1_t, (float)b2_t);
            timed_run(weight_updates_kernel, "weight_updates");
        }
        correct.read_from_device();
        loss.read_from_device();
        epoch_loss += loss[0];
        epoch_correct += (size_t)correct[0];

        double avg_train_loss = epoch_loss / (double)train.observations;
        double avg_train_acc  = epoch_correct / (double)train.observations;

        if (avg_train_loss < best_loss) {
            best_loss = avg_train_loss;
        }

        if (avg_train_acc > best_acc) {
            best_acc = avg_train_acc;
        }

        // ---- Test evaluation (forward + loss only) ----
        double epoch_test_loss    = 0.0;
        size_t epoch_test_correct = 0;

        if (test.observations > 0) {
            correct.reset();
            loss.reset();

            // Swap encode_kernel data buffer to test_data (param pos 1)
            encode_kernel.set_parameters(1, test_data);

            for (int obs = 0; obs < (int)test.observations; obs++) {
                x.reset();
                v.reset();
                s.reset();
                v_pre.reset();
                dL_ds.reset();

                // Encode test observation
                encode_kernel.set_parameters(2, (int)test.cols);
                encode_kernel.set_parameters(5, (uint)obs);
                timed_run(encode_kernel, "encode");

                // Forward pass
                for (size_t t = 0; t < nc.timesteps; t++) {
                    forward_kernel.set_parameters(14, (uint)t);
                    timed_run(forward_kernel, "forward");
                }

                // Loss
                loss_kernel.set_parameters(7, (uint)(test.labels[obs]));
                timed_run(loss_kernel, "loss");
            }

            // We only need to read these once an epoch, cuts down on memory
            // bandwidth
            correct.read_from_device();
            loss.read_from_device();
            epoch_test_correct += (size_t)correct[0];
            epoch_test_loss += loss[0];

            // Restore encode_kernel data buffer to train data
            encode_kernel.set_parameters(1, data);
        }

        double avg_test_loss = test.observations > 0
                                   ? epoch_test_loss / (double)test.observations
                                   : 0.0;
        double avg_test_acc =
            test.observations > 0
                ? epoch_test_correct / (double)test.observations
                : 0.0;

        if (test.observations > 0) {
            if (avg_test_loss < best_test_loss) {
                best_test_loss = avg_test_loss;
            }
            if (avg_test_acc > best_test_acc) {
                best_test_acc = avg_test_acc;
            }
        }

        printf("Epoch: %4zu/%zu, Loss: %10g / %10g, Acc: %10g / %10g, Test "
               "Loss: %10g / %10g, Test Acc: %10g / %10g\n",
               epoch + 1, epochs, avg_train_loss, best_loss, avg_train_acc,
               best_acc, avg_test_loss, best_test_loss, avg_test_acc,
               best_test_acc);

        /* Periodic CPU eval: read GPU weights back, run CPU forward+loss on
         * test */
        if (cfg.cpu_eval_interval > 0 &&
            (epoch + 1) % cfg.cpu_eval_interval == 0) {
            weights.read_from_device();
            std::vector<std::vector<double>> cpu_weights;
            read_gpu_weights(weights, nc.total_neurons, max_incoming,
                             state->weights, nc.scale_factor, cpu_weights);
            write_weights_to_network(n, nc.total_neurons, nc.discrete,
                                     nc.scale_factor, cpu_weights);
            std::pair<double, double> cpu_result =
                cpu_eval_test(n, nc, test, cpu_weights, state->delays,
                              state->thresholds, rho, tau);
            printf(
                "  [CPU eval @ epoch %4zu] Test Loss: %10g, Test Acc: %10g\n",
                epoch + 1, cpu_result.first, cpu_result.second);
        }
    }

    auto t_end = chrono::high_resolution_clock::now();
    double total_us =
        chrono::duration<double, std::micro>(t_end - t_start).count();
    timing_print(total_us);

    /* Final: read GPU weights, CPU eval, export network JSON */
    weights.read_from_device();
    read_gpu_weights(weights, nc.total_neurons, max_incoming, state->weights,
                     nc.scale_factor, state->weights);

    /* CPU final test eval */
    write_weights_to_network(n, nc.total_neurons, nc.discrete, nc.scale_factor,
                             state->weights);
    std::pair<double, double> final_result =
        cpu_eval_test(n, nc, test, state->weights, state->delays,
                      state->thresholds, rho, tau);
    printf("Final CPU Test Loss: %10g, Final CPU Test Acc: %10g\n",
           final_result.first, final_result.second);

    /* Export if requested */
    if (export_json && !cfg.network_json_out.empty()) {
        export_network_json(n, cfg.network_json_out);
    }

    return true;
}
