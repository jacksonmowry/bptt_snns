#include "opencl_backend.h"
#include "forward_backward.h"
#include "network_utils.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace std;
using namespace neuro;

struct KernelTiming {
    string name;
    double total_us;
    double min_us;
    double max_us;
    uint64_t calls;
    KernelTiming()
        : name(""), total_us(0.0), min_us(1e18), max_us(0.0), calls(0) {}
    KernelTiming(const string& n, double u, uint64_t c)
        : name(n), total_us(u), min_us(u), max_us(u), calls(c) {}
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
            kt.min_us = std::min(kt.min_us, us);
            kt.max_us = std::max(kt.max_us, us);
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

static void encode(Memory<double>& data, const Dataset& d, bool timeseries) {
    if (timeseries) {
        // data = [observations * (input_features * 2) * dataset_timesteps]
        assert(data.length() ==
               (unsigned long)(d.shape[0] * (d.shape[1] * 2) * d.shape[2]));

        for (int obs = 0; obs < d.shape[0]; obs++) {
            for (int input_feature = 0; input_feature < d.shape[1];
                 input_feature++) {
                double range =
                    d.max_vals[input_feature] - d.min_vals[input_feature];
                if (range == 0.0) {
                    continue;
                }

                for (int dataset_timestep = 0; dataset_timestep < d.shape[2];
                     dataset_timestep++) {
                    double x     = (d.data[(obs * d.shape[1] * d.shape[2]) +
                                           (input_feature * d.shape[2]) +
                                           (dataset_timestep)] -
                                    d.min_vals[input_feature]) /
                                   range;
                    double inv_x = 1.0 - x;

                    if (x > 0.0) {
                        size_t idx = (obs * (d.shape[1] * 2) * d.shape[2]) +
                                     (input_feature * 2 * d.shape[2]) +
                                     (dataset_timestep);
                        assert(idx < (size_t)(d.shape[0] * (d.shape[1] * 2) *
                                              d.shape[2]));
                        data[(obs * (d.shape[1] * 2) * d.shape[2]) +
                             (input_feature * 2 * d.shape[2]) +
                             (dataset_timestep)] = 1.0 / x;
                    }

                    if (inv_x > 0.0) {
                        size_t idx = (obs * (d.shape[1] * 2) * d.shape[2]) +
                                     ((input_feature * 2 + 1) * d.shape[2]) +
                                     (dataset_timestep);
                        assert(idx < (size_t)(d.shape[0] * (d.shape[1] * 2) *
                                              d.shape[2]));
                        data[(obs * (d.shape[1] * 2) * d.shape[2]) +
                             ((input_feature * 2 + 1) * d.shape[2]) +
                             (dataset_timestep)] = 1.0 / inv_x;
                    }
                }
            }
        }
    } else {
        // data = [observations * (input_features * 2)]
        assert(data.length() == (unsigned long)(d.shape[0] * (d.shape[1] * 2)));

        for (int row = 0; row < d.shape[0]; row++) {
            for (int col = 0; col < d.shape[1]; col++) {
                double range = d.max_vals[col] - d.min_vals[col];
                if (range == 0.0) {
                    continue;
                }

                double x =
                    (d.data[row * d.shape[1] + col] - d.min_vals[col]) / range;
                double inv_x = 1.0 - x;

                if (x > 0.0) {
                    data[(row * d.shape[1] * 2) + (col * 2)] = double(1.0 / x);
                }
                if (inv_x > 0.0) {
                    data[(row * d.shape[1] * 2) + (col * 2 + 1)] =
                        double(1.0 / inv_x);
                }
            }
        }
    }
}

static void write_weights_to_network(neuro::Network* n, size_t total_neurons,
                                     Memory<short>& weights,
                                     size_t max_incoming) {
    for (size_t i = 0; i < total_neurons; i++) {
        auto& incoming = n->get_node(i)->incoming;
        for (size_t j = 0; j < incoming.size() && j < max_incoming; j++) {
            incoming[j]->set("Weight", weights[(i * max_incoming) + j]);
        }
    }
}

static std::pair<double, double> cpu_eval_test(neuro::Network* n,
                                               const NetworkConfiguration& nc,
                                               const Dataset& d, double rho,
                                               double tau) {
    if (d.shape[0] == 0) {
        return {0.0, 0.0};
    }

    TrainingBundle tb(nc.total_neurons, nc.timesteps, nc.output_neurons, rho,
                      tau, {}, {}, {});
    neuro::Processor* p = nullptr;
    load_network(&p, n);

    double total_loss    = 0.0;
    size_t total_correct = 0;

    for (int obs = 0; obs < d.shape[0]; obs++) {
        EvaluationResults er = forward(&tb, p, &d, (size_t)obs, &nc);
        total_loss += er.loss;
        total_correct += (size_t)er.correct;
    }

    delete p;
    return {total_loss / d.shape[0], (double)total_correct / d.shape[0]};
}

OpenclBackend::OpenclBackend(const CliConfig& cfg, NetworkConfiguration& nc,
                             const Dataset& train, const Dataset& test,
                             size_t max_incoming, size_t max_outgoing)
    : cfg(cfg), nc(nc), train(train), test(test), max_incoming(max_incoming),
      max_outgoing(max_outgoing), batch_size(cfg.batch_size),
      learning_rate(cfg.learning_rate), decay_rate(cfg.decay_rate),
      rho(cfg.rho), tau(cfg.tau), b1_t(1.0), b2_t(1.0) {

    Device device(select_device_with_most_flops());
    const size_t encode_work_size        = nc.input_neurons;
    const size_t forward_work_size       = nc.total_neurons;
    const size_t loss_work_size          = nc.output_neurons;
    const size_t backward_grad_work_size = nc.total_neurons;
    const size_t backward_delta_w_work_size =
        nc.total_neurons * nc.max_incoming;
    const size_t weight_updates_work_size = nc.total_neurons * nc.max_incoming;

    x.reset(new Memory<short>(device, nc.input_neurons * nc.timesteps));

    data.reset(new Memory<double>(
        device, cfg.timeseries
                    ? train.shape[0] * (train.shape[1] * 2) * train.shape[2]
                    : train.shape[0] * train.shape[1] * 2));

    if (test.shape[0] > 0) {
        test_data.reset(new Memory<double>(
            device, cfg.timeseries
                        ? test.shape[0] * (test.shape[1] * 2) * test.shape[2]
                        : test.shape[0] * test.shape[1] * 2));
    }

    v_thresh.reset(new Memory<short>(device, nc.total_neurons));
    weights.reset(
        new Memory<short>(device, nc.total_neurons * nc.max_incoming));
    delays.reset(new Memory<uint>(device, nc.total_neurons * nc.max_incoming));
    incoming.reset(new Memory<uint>(device, nc.total_neurons));
    incoming_ids.reset(
        new Memory<uint>(device, nc.total_neurons * nc.max_incoming));
    is_input_neuron.reset(new Memory<uchar>(device, nc.total_neurons));
    is_output_neuron.reset(new Memory<uchar>(device, nc.total_neurons));
    v.reset(new Memory<int>(device, nc.total_neurons));
    s.reset(new Memory<char>(device, nc.total_neurons * nc.timesteps));
    v_pre.reset(new Memory<int>(device, nc.total_neurons * nc.timesteps));
    dL_ds.reset(new Memory<float>(device, nc.output_neurons));
    correct.reset(new Memory<float>(device, 1));
    loss.reset(new Memory<float>(device, 1));
    spike_grad_history.reset(
        new Memory<float>(device, nc.total_neurons * nc.timesteps));
    future_mem_grad.reset(new Memory<float>(device, nc.total_neurons));
    delta_W.reset(
        new Memory<float>(device, nc.total_neurons * nc.max_incoming));
    neuron_grad.reset(new Memory<float>(device, nc.total_neurons));
    m_weights.reset(
        new Memory<float>(device, nc.total_neurons * nc.max_incoming));
    v_weights.reset(
        new Memory<float>(device, nc.total_neurons * nc.max_incoming));
    outgoing.reset(new Memory<uint>(device, nc.total_neurons));
    gradient_slot.reset(
        new Memory<uint>(device, nc.total_neurons * nc.max_incoming));
    gradient_accumulators.reset(new Memory<float>(
        device, nc.timesteps * nc.total_neurons * nc.max_outgoing));

    encode_kernel.reset(
        new Kernel(device, encode_work_size, "risp_encode_inputs_kernel", *x,
                   *data, (int)train.shape[1], (int)nc.input_neurons,
                   (int)nc.timesteps, (uint)0, (short)nc.spike_value_factor));

    encode_timeseries_kernel.reset(new Kernel(
        device, encode_work_size, "risp_encode_timeseries_inputs_kernel", *x,
        *data, (int)train.shape[2], (int)nc.input_neurons, (int)nc.timesteps,
        (uint)0, (short)nc.spike_value_factor));

    forward_kernel.reset(new Kernel(
        device, forward_work_size, "risp_forward_kernel", *x, *v_thresh,
        *weights, *delays, *incoming, *incoming_ids, *is_input_neuron, *v, *s,
        *v_pre, (short)nc.leak, (int)(nc.min_potential / nc.scale_factor),
        (uint)nc.total_neurons, (uint)nc.timesteps, (uint)0,
        (uint)nc.max_incoming));

    loss_kernel.reset(
        new Kernel(device, loss_work_size, "risp_loss_kernel", *s, *dL_ds,
                   *correct, *loss, (uint)nc.total_neurons,
                   (uint)nc.output_neurons, (uint)nc.timesteps, (uint)0));

    backward_grad_kernel.reset(new Kernel(
        device, backward_grad_work_size, "risp_backward_grad_kernel", *dL_ds,
        *s, *v_pre, *v_thresh, *gradient_accumulators, *outgoing,
        *is_output_neuron, *spike_grad_history, *future_mem_grad, *neuron_grad,
        (short)nc.leak, (float)nc.min_potential, (float)cfg.tau, (float)cfg.rho,
        (uint)nc.total_neurons, (uint)nc.output_neurons, (short)nc.timesteps,
        (float)nc.scale_factor, (uint)nc.max_outgoing, (short)0));

    backward_delta_w_kernel.reset(new Kernel(
        device, backward_delta_w_work_size, "risp_backward_delta_w_kernel",
        *neuron_grad, *s, *weights, *delays, *incoming, *incoming_ids,
        *gradient_slot, *spike_grad_history, *delta_W, *gradient_accumulators,
        (uint)nc.total_neurons, (uint)nc.max_incoming, (uint)nc.max_outgoing,
        (short)nc.timesteps, (float)nc.scale_factor, (short)0));

    weight_updates_kernel.reset(new Kernel(
        device, weight_updates_work_size, "weight_updates_kernel", *incoming,
        *m_weights, *v_weights, *delta_W, *weights, (uint)nc.total_neurons,
        (uint)nc.max_incoming, (float)cfg.learning_rate, (float)cfg.decay_rate,
        (uint)1, (uint)cfg.batch_size, (uint)0, (uint)0, (float)0.9f,
        (float)0.999f, (float)0.0f, (float)0.0f, (uint)nc.timesteps,
        (uint)train.shape[0], (float)nc.scale_factor, (short)nc.min_weight,
        (short)nc.max_weight, (int)nc.steps));

    // Encode data
    encode(*data, train, cfg.timeseries);
    if (test.shape[0] > 0) {
        encode(*test_data, test, cfg.timeseries);
    }

    // Initialize GPU buffers from network
    for (size_t i = 0; i < nc.total_neurons; i++) {
        neuro::Node* node = nc.n->get_node(i);

        (*v_thresh)[i]         = (short)node->get("Threshold");
        (*incoming)[i]         = node->incoming.size();
        (*is_input_neuron)[i]  = i < nc.input_neurons;
        (*is_output_neuron)[i] = i >= nc.input_neurons + nc.hidden_neurons;

        for (size_t j = 0; j < node->incoming.size(); j++) {
            neuro::Edge* edge  = node->incoming[j];
            size_t incoming_id = edge->from->id;

            (*weights)[(i * max_incoming) + j]      = edge->get("Weight");
            (*delays)[(i * max_incoming) + j]       = edge->get("Delay");
            (*incoming_ids)[(i * max_incoming) + j] = edge->from->id;

            (*gradient_slot)[i * nc.max_incoming + j] =
                (*outgoing)[incoming_id];
            (*outgoing)[incoming_id]++;
        }
    }

    m_weights->reset();
    v_weights->reset();

    data->write_to_device();
    if (test.shape[0] > 0) {
        test_data->write_to_device();
    }
    v_thresh->write_to_device();
    weights->write_to_device();
    delays->write_to_device();
    incoming->write_to_device();
    incoming_ids->write_to_device();
    is_input_neuron->write_to_device();
    is_output_neuron->write_to_device();
    m_weights->write_to_device();
    v_weights->write_to_device();
    outgoing->write_to_device();
    gradient_slot->write_to_device();

    // Init batch order
    batch_order.resize(train.shape[0]);
    for (int i = 0; i < train.shape[0]; i++) {
        batch_order[i] = (size_t)i;
    }

    timing_start(cfg.opencl_timings);
    t_start = chrono::high_resolution_clock::now();
}

void OpenclBackend::do_one_epoch(size_t epoch) {
    double epoch_loss    = 0.0;
    size_t epoch_correct = 0;
    correct->reset();
    loss->reset();

    // Shuffle batch order each epoch
    for (int i = 0; i < train.shape[0]; i++) {
        int j          = rand() % train.shape[0];
        size_t tmp     = batch_order[i];
        batch_order[i] = batch_order[j];
        batch_order[j] = tmp;
    }

    // Mini-batch SGD loop
    for (int batch_start = 0; batch_start < train.shape[0];
         batch_start += (int)batch_size) {
        size_t current_batch_size =
            min(batch_size, (size_t)(train.shape[0] - batch_start));

        // Reset accumulators for this batch
        delta_W->reset();

        for (size_t b = 0; b < current_batch_size; b++) {
            size_t obs = batch_order[(size_t)batch_start + b];

            x->reset();
            v->reset();
            s->reset();
            v_pre->reset();
            dL_ds->reset();
            spike_grad_history->reset();
            future_mem_grad->reset();
            gradient_accumulators->reset();

            // Encode data
            if (cfg.timeseries) {
                encode_timeseries_kernel->set_parameters(5, (uint)obs);
                timed_run(*encode_timeseries_kernel, "encode_timeseries");
            } else {
                encode_kernel->set_parameters(5, (uint)obs);
                timed_run(*encode_kernel, "encode");
            }

            // Forward pass
            for (size_t t = 0; t < nc.timesteps; t++) {
                forward_kernel->set_parameters(14, (uint)t);
                timed_run(*forward_kernel, "forward");
            }

            // Loss
            loss_kernel->set_parameters(7, (uint)(train.labels[obs]));
            timed_run(*loss_kernel, "loss");

            // Backwards
            for (short t = nc.timesteps - 1; t >= 0; t--) {
                backward_grad_kernel->set_parameters(19, (short)t);
                timed_run(*backward_grad_kernel, "backward_grad");
                backward_delta_w_kernel->set_parameters(15, (short)t);
                timed_run(*backward_delta_w_kernel, "backward_delta_w");
            }
        }

        // Weight updates
        b1_t *= 0.9;
        b2_t *= 0.999;
        weight_updates_kernel->set_parameters(9, (uint)current_batch_size,
                                              (uint)batch_size);
        weight_updates_kernel->set_parameters(11, (uint)batch_start);
        weight_updates_kernel->set_parameters(12, (uint)epoch);
        weight_updates_kernel->set_parameters(15, (float)b1_t, (float)b2_t);
        timed_run(*weight_updates_kernel, "weight_updates");
    }

    correct->read_from_device();
    loss->read_from_device();
    epoch_loss += (*loss)[0];
    epoch_correct += (size_t)(*correct)[0];

    double avg_train_loss = epoch_loss / (double)train.shape[0];
    double avg_train_acc  = epoch_correct / (double)train.shape[0];

    // Test evaluation
    double epoch_test_loss    = 0.0;
    size_t epoch_test_correct = 0;

    if (test.shape[0] > 0) {
        correct->reset();
        loss->reset();

        if (cfg.timeseries) {
            encode_timeseries_kernel->set_parameters(1, *test_data);
        } else {
            encode_kernel->set_parameters(1, *test_data);
        }

        for (int obs = 0; obs < (int)test.shape[0]; obs++) {
            x->reset();
            v->reset();
            s->reset();
            v_pre->reset();
            dL_ds->reset();

            if (cfg.timeseries) {
                encode_timeseries_kernel->set_parameters(2, (int)test.shape[2]);
                encode_timeseries_kernel->set_parameters(5, (uint)obs);
                timed_run(*encode_timeseries_kernel, "encode_timeseries");
            } else {
                encode_kernel->set_parameters(2, (int)test.shape[1]);
                encode_kernel->set_parameters(5, (uint)obs);
                timed_run(*encode_kernel, "encode");
            }

            for (size_t t = 0; t < nc.timesteps; t++) {
                forward_kernel->set_parameters(14, (uint)t);
                timed_run(*forward_kernel, "forward");
            }

            loss_kernel->set_parameters(7, (uint)(test.labels[obs]));
            timed_run(*loss_kernel, "loss");
        }

        correct->read_from_device();
        loss->read_from_device();
        epoch_test_correct += (size_t)(*correct)[0];
        epoch_test_loss += (*loss)[0];

        if (cfg.timeseries) {
            encode_timeseries_kernel->set_parameters(1, *data);
        } else {
            encode_kernel->set_parameters(1, *data);
        }
    }

    double avg_test_loss =
        test.shape[0] > 0 ? epoch_test_loss / (double)test.shape[0] : 0.0;
    double avg_test_acc =
        test.shape[0] > 0 ? epoch_test_correct / (double)test.shape[0] : 0.0;

    stats.train_acc  = avg_train_acc;
    stats.train_loss = avg_train_loss;
    stats.test_acc   = avg_test_acc;
    stats.test_loss  = avg_test_loss;

    // Periodic CPU eval
    if (cfg.cpu_eval_interval > 0 && (epoch + 1) % cfg.cpu_eval_interval == 0) {
        weights->read_from_device();
        write_weights_to_network(nc.n, nc.total_neurons, *weights,
                                 nc.max_incoming);
        std::pair<double, double> cpu_result =
            cpu_eval_test(nc.n, nc, test.shape[0] > 0 ? test : train, rho, tau);

        if (test.shape[0] > 0) {
            printf(
                "  [CPU eval @ epoch %4zu] Test Loss: %10g, Test Acc: %10g\n",
                epoch + 1, cpu_result.first, cpu_result.second);
        } else {
            printf(
                "  [CPU eval @ epoch %4zu] Train Loss: %10g, Train Acc: %10g\n",
                epoch + 1, cpu_result.first, cpu_result.second);
        }
    }
}

OpenclBackend::~OpenclBackend() {
    auto t_end = chrono::high_resolution_clock::now();
    double total_us =
        chrono::duration<double, std::micro>(t_end - t_start).count();
    timing_print(total_us);

    // Read GPU weights back to host for state
    weights->read_from_device();
    write_weights_to_network(nc.n, nc.total_neurons, *weights, nc.max_incoming);

    // Final CPU eval on GPU weights
    std::pair<double, double> cpu_result =
        cpu_eval_test(nc.n, nc, test.shape[0] > 0 ? test : train, rho, tau);
    if (test.shape[0] > 0) {
        printf("Final CPU Test Loss: %10g, Final CPU Test Acc: %10g\n",
               cpu_result.first, cpu_result.second);
    } else {
        printf("Final CPU Train Loss: %10g, Final CPU Train Acc: %10g\n",
               cpu_result.first, cpu_result.second);
    }
}

TrainingStats OpenclBackend::get_stats() const { return stats; }

void OpenclBackend::update_weights(neuro::Network* network) {
    // Read GPU weights back to host — all weight mutation happens on GPU
    m_weights->read_from_device();
    write_weights_to_network(network, nc.total_neurons, *weights,
                             nc.max_incoming);
}
