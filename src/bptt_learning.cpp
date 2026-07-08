#include "cli.h"
#include "data_utils.h"
#include "forward_backward.h"
#include "framework.hpp"
#include "math_utils.h"
#include "network_setup.h"
#include "network_utils.h"
#include "opencl.hpp"
#include "optimizer.h"
#include "shared.h"
#include "threading.h"
#include "training.h"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;
using namespace neuro;

void encode(Memory<double>& data, const Dataset& d) {
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

int main(int argc, char* argv[]) {
    CliConfig cfg;
    int rc = parse_cli(argc, argv, &cfg);
    if (rc != 0) {
        print_usage(argv[0]);
        return 1;
    }
    if (cfg.show_help) {
        print_usage(argv[0]);
        return 0;
    }
    if (cfg.network_json_file.empty()) {
        fprintf(stderr, "Error: --network_json is required\n");
        print_usage(argv[0]);
        return 1;
    }

    bool have_simple = !cfg.data_file.empty() && !cfg.label_file.empty();
    bool have_split =
        !cfg.train_data_file.empty() && !cfg.train_label_file.empty() &&
        !cfg.test_data_file.empty() && !cfg.test_label_file.empty();
    if (have_simple && have_split) {
        fprintf(stderr, "Error: cannot specify both (-d + -l) and "
                        "(-a + -i + -j + -k); choose one\n");
        print_usage(argv[0]);
        return 1;
    }
    if (!have_simple && !have_split) {
        fprintf(stderr, "Error: either (-d + -l) OR "
                        "(-a + -i + -j + -k) are required\n");
        print_usage(argv[0]);
        return 1;
    }

    srand(cfg.seed);
    srand48(cfg.seed);

    Dataset train;
    Dataset test;
    if (cfg.timeseries) {
        assert(false);
        load_dataset_2d(cfg.data_file.c_str(), cfg.label_file.c_str(),
                        cfg.training_percent, &train, &test);
    } else if (have_simple) {
        load_dataset(cfg.data_file.c_str(), cfg.label_file.c_str(),
                     cfg.training_percent, &train, &test);
    } else {
        load_dataset_single(cfg.train_data_file.c_str(),
                            cfg.train_label_file.c_str(), &train);
        load_dataset_single(cfg.test_data_file.c_str(),
                            cfg.test_label_file.c_str(), &test);
    }

    size_t train_labels = label_count(&train);
    size_t test_labels  = label_count(&test);
    assert(test.observations == 0 || train_labels == test_labels);

    size_t input_neurons =
        (cfg.timeseries) ? train.rows_per_observation * 2 : train.cols * 2;
    size_t output_neurons = train_labels;
    size_t hidden_neurons = cfg.hidden_neurons;
    size_t total_neurons  = input_neurons + hidden_neurons + output_neurons;

    double connectivity  = cfg.connectivity;
    double learning_rate = cfg.learning_rate;
    double decay_rate    = cfg.decay_rate;
    double tau           = cfg.tau;
    double rho           = cfg.rho;
    size_t timesteps     = cfg.timesteps;
    unsigned long seed   = cfg.seed;
    size_t epochs        = cfg.epochs;
    size_t batch_size    = cfg.batch_size;
    double training_pct  = cfg.training_percent;
    size_t threads       = cfg.threads;
    bool timeseries      = cfg.timeseries;

    Network* n = load_and_init_network(
        cfg.network_json_file, connectivity, learning_rate, decay_rate, tau,
        rho, timesteps, hidden_neurons, seed, epochs, batch_size, training_pct,
        threads, timeseries);

    bool discrete         = n->get_data("proc_params")["discrete"];
    std::string leak_prop = n->get_data("proc_params")["leak_mode"];
    bool leak             = leak_prop == "all";
    double min_potential  = n->get_data("proc_params")["min_potential"];
    double min_weight     = n->get_data("proc_params")["min_weight"];
    double max_weight     = n->get_data("proc_params")["max_weight"];
    double spike_value_factor =
        n->get_data("proc_params")["spike_value_factor"];
    double max_threshold = n->get_data("proc_params")["max_threshold"];
    int scale            = 0;
    if (discrete) {
        scale = max(abs(min_weight), abs(max_weight)) * 2 + 1;
        scale = pow(2.0, ceil(log2(scale)));
    }
    double scale_factor = 2.0 / scale;
    if (discrete) {
        min_potential *= scale_factor;
    }

    size_t neuron_count, synapse_count;
    if (n->num_nodes() == 0) {
        std::tie(neuron_count, synapse_count) = generate_network(
            n, input_neurons, hidden_neurons, output_neurons, total_neurons,
            connectivity, discrete, scale, scale_factor, min_weight, max_weight,
            max_threshold);
        printf("Neurons: %zu, Synapses: %zu\n", neuron_count, synapse_count);
    } else {
        neuron_count  = n->num_nodes();
        synapse_count = n->num_edges();
        printf("Resuming training with Neurons: %zu, Synapses: %zu\n",
               neuron_count, synapse_count);
    }
    n->make_sorted_node_vector();

    build_run_metadata(
        n, argc, argv, cfg, input_neurons, output_neurons, total_neurons,
        neuron_count, synapse_count, discrete, min_potential, min_weight,
        max_weight, max_threshold, leak_prop, scale, scale_factor, connectivity,
        learning_rate, decay_rate, tau, rho, timesteps, hidden_neurons, seed,
        epochs, batch_size, training_pct, threads, timeseries);

    NetworkConfiguration nc = {
        .n              = n,
        .input_neurons  = input_neurons,
        .hidden_neurons = hidden_neurons,
        .output_neurons = output_neurons,
        .layer_offsets  = {0, input_neurons, input_neurons + hidden_neurons},
        .total_neurons  = total_neurons,
        .max_incoming   = 0,
        .timesteps      = timesteps,
        .timeseries     = timeseries,
        .min_potential  = min_potential,
        .leak           = leak,
        .scale_factor   = scale_factor,
        .steps          = scale,
        .discrete       = discrete,
        .min_weight     = min_weight,
        .max_weight     = max_weight,
        .spike_value_factor = spike_value_factor,
    };

    TrainingState* state = init_training(n, nc, train, threads, rho, tau);

    init_network_weights(n, total_neurons, discrete, scale_factor,
                         state->weights, state->delays, state->thresholds);

    size_t max_incoming = 0;
    for (size_t i = 0; i < state->weights.size(); i++) {
        if (state->weights[i].size() > max_incoming) {
            max_incoming = state->weights[i].size();
        }
    }
    nc.max_incoming = max_incoming;

    if (cfg.opencl) {
        if (!discrete) {
            fprintf(
                stderr,
                "OpenCL support is not enabled for non-discrete networks.\n");
            exit(1);
        }

        Device device(select_device_with_most_flops());
        const size_t encode_work_size   = nc.input_neurons;
        const size_t forward_work_size  = nc.total_neurons;
        const size_t loss_work_size     = nc.output_neurons;
        const size_t backward_work_size = nc.total_neurons * nc.max_incoming;
        const size_t weight_updates_work_size =
            nc.total_neurons * nc.max_incoming;

        Memory<short> x(device, nc.input_neurons * nc.timesteps);
        Memory<double> data(device, train.observations * train.cols * 2);
        Memory<double> test_data(device, test.observations * test.cols * 2);
        Memory<short> v_thresh(device, nc.total_neurons);
        Memory<short> weights(device, nc.total_neurons * nc.max_incoming);
        Memory<ushort> delays(device, nc.total_neurons * nc.max_incoming);
        Memory<ushort> incoming(device, nc.total_neurons);
        Memory<ushort> incoming_ids(device, nc.total_neurons * nc.max_incoming);
        Memory<uchar> is_input_neuron(device, nc.total_neurons);
        Memory<uchar> is_output_neuron(device, nc.total_neurons);
        Memory<short> v(device, nc.total_neurons);
        Memory<char> s(device, nc.total_neurons * nc.timesteps);
        Memory<short> v_pre(device, nc.total_neurons * nc.timesteps);
        Memory<float> dL_ds(device, nc.output_neurons);
        Memory<float> correct(device, 1);
        Memory<float> loss(device, 1);
        Memory<float> spike_grad_history(device,
                                         nc.total_neurons * nc.timesteps);
        Memory<float> voltage_grad_history(device,
                                           nc.total_neurons * nc.timesteps);
        Memory<float> future_mem_grad(device, nc.total_neurons);
        Memory<float> delta_W(device, nc.total_neurons * nc.max_incoming);
        Memory<float> m_weights(device, nc.total_neurons * nc.max_incoming);
        Memory<float> v_weights(device, nc.total_neurons * nc.max_incoming);

        Kernel encode_kernel(device, encode_work_size,
                             "risp_encode_inputs_kernel", x, data, (int)train.cols,
                             (int)nc.input_neurons, (int)nc.timesteps, (uint)0,
                             (short)nc.spike_value_factor);

        Kernel forward_kernel(
            device, forward_work_size, "risp_forward_kernel", x, v_thresh,
            weights, delays, incoming, incoming_ids, is_input_neuron, v, s,
            v_pre, (short)nc.leak, (short)(nc.min_potential / nc.scale_factor),
            (ushort)nc.total_neurons, (ushort)nc.timesteps, (ushort)0,
            (ushort)nc.max_incoming);

        Kernel loss_kernel(device, loss_work_size, "risp_loss_kernel", s, dL_ds,
                           correct, loss, (ushort)nc.total_neurons,
                           (ushort)nc.output_neurons, (ushort)nc.timesteps,
                           (ushort)0);

        Kernel backward_kernel(
            device, backward_work_size, "risp_backward_kernel", dL_ds, s, v_pre,
            v_thresh, is_output_neuron, weights, delays, incoming, incoming_ids,
            spike_grad_history, voltage_grad_history, future_mem_grad, delta_W,
            (short)nc.leak, (float)nc.min_potential, (float)tau, (float)rho,
            (ushort)nc.total_neurons, (ushort)nc.output_neurons,
            (short)nc.timesteps, (ushort)nc.max_incoming,
            (float)nc.scale_factor, (short)0);

        Kernel weight_updates_kernel(
            device, weight_updates_work_size, "weight_updates_kernel", incoming,
            m_weights, v_weights, delta_W, weights, (ushort)nc.total_neurons,
            (ushort)nc.max_incoming, (float)learning_rate, (float)decay_rate,
            (ushort)1, (ushort)batch_size, (uint)0, (uint)0, (float)0.9f,
            (float)0.999f, (float)0.0f, (float)0.0f, (ushort)nc.timesteps,
            (uint)train.observations, (float)nc.scale_factor,
            (short)nc.min_weight, (short)nc.max_weight, (int)nc.steps);

        // Temporary handling for numeric instability
        encode(data, train);
        encode(test_data, test);

        for (size_t i = 0; i < nc.total_neurons; i++) {
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

        double b1_t = 1.0;
        double b2_t = 1.0;

        // Shuffle order for SGD
        vector<size_t> batch_order(train.observations);
        for (int i = 0; i < train.observations; i++) {
            batch_order[i] = (size_t)i;
        }

        double best_acc = 0.0;
        double best_loss = DBL_MAX;
        double best_test_acc = 0.0;
        double best_test_loss = DBL_MAX;

        for (size_t epoch = 0; epoch < cfg.epochs; epoch++) {
            double epoch_loss = 0.0;
            size_t epoch_correct = 0;

            // Shuffle batch order each epoch
            for (int i = 0; i < train.observations; i++) {
                int j = rand() % train.observations;
                size_t tmp = batch_order[i];
                batch_order[i] = batch_order[j];
                batch_order[j] = tmp;
            }

            // Mini-batch SGD loop
            for (int batch_start = 0; batch_start < train.observations;
                 batch_start += (int)batch_size) {
                size_t current_batch_size = min(
                    batch_size, (size_t)(train.observations - batch_start));

                // Reset accumulators for this batch
                delta_W.reset();
                correct.reset();
                loss.reset();

                for (size_t b = 0; b < current_batch_size; b++) {
                    size_t obs = batch_order[(size_t)batch_start + b];

                    x.reset();
                    v.reset();
                    s.reset();
                    v_pre.reset();
                    dL_ds.reset();
                    spike_grad_history.reset();
                    voltage_grad_history.reset();
                    future_mem_grad.reset();

                    // Encode data
                    encode_kernel.set_parameters(5, (uint)obs);
                    encode_kernel.run();

                    // Forward pass
                    for (size_t t = 0; t < nc.timesteps; t++) {
                        forward_kernel.set_parameters(14, (ushort)t);
                        forward_kernel.run();
                    }

                    // Loss
                    loss_kernel.set_parameters(
                        7, (ushort)(train.labels[obs]));
                    loss_kernel.run();

                    // Backwards
                    for (short t = nc.timesteps - 1; t >= 0; t--) {
                        backward_kernel.set_parameters(22, (short)t);
                        backward_kernel.run();
                    }
                }

                // Weight updates
                // 9 current_batch_size, 10 batch_size
                // 11 batch_start, 12 epoch
                // 15-16 b1_t b2_t
                b1_t *= 0.9;
                b2_t *= 0.999;
                weight_updates_kernel.set_parameters(
                    9, (ushort)current_batch_size, (ushort)batch_size);
                weight_updates_kernel.set_parameters(11, (uint)batch_start);
                weight_updates_kernel.set_parameters(12, (uint)epoch);
                weight_updates_kernel.set_parameters(15, (float)b1_t, (float)b2_t);
                weight_updates_kernel.run();

                correct.read_from_device();
                loss.read_from_device();
                epoch_loss += loss[0];
                epoch_correct += (size_t)correct[0];
            }

            double avg_train_loss = epoch_loss / (double)train.observations;
            double avg_train_acc  = epoch_correct / (double)train.observations;

            if (avg_train_loss < best_loss) {
                best_loss = avg_train_loss;
            }

            if (avg_train_acc > best_acc) {
                best_acc = avg_train_acc;
            }
            // ---- Test evaluation (forward + loss only) ----
            double epoch_test_loss = 0.0;
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
                    encode_kernel.run();

                    // Forward pass
                    for (size_t t = 0; t < nc.timesteps; t++) {
                        forward_kernel.set_parameters(14, (ushort)t);
                        forward_kernel.run();
                    }

                    // Loss
                    loss_kernel.set_parameters(
                        7, (ushort)(test.labels[obs]));
                    loss_kernel.run();

                }

                correct.read_from_device();
                loss.read_from_device();
                epoch_test_correct += (size_t)correct[0];
                epoch_test_loss += loss[0];

                // Restore encode_kernel data buffer to train data
                encode_kernel.set_parameters(1, data);
            }

            double avg_test_loss = test.observations > 0
                ? epoch_test_loss / (double)test.observations : 0.0;
            double avg_test_acc  = test.observations > 0
                ? epoch_test_correct / (double)test.observations : 0.0;

            if (test.observations > 0) {
                if (avg_test_loss < best_test_loss) {
                    best_test_loss = avg_test_loss;
                }
                if (avg_test_acc > best_test_acc) {
                    best_test_acc = avg_test_acc;
                }
            }

            printf("Epoch: %4zu/%zu, Loss: %10g / %10g, Acc: %10g / %10g, Test Loss: %10g / %10g, Test Acc: %10g / %10g\n",
                   epoch + 1, cfg.epochs, avg_train_loss, best_loss, avg_train_acc, best_acc, avg_test_loss, best_test_loss, avg_test_acc, best_test_acc);
        }

        exit(0);
    } else {
        run_training(cfg, n, nc, train, test, state, epochs, batch_size,
                     learning_rate, decay_rate);
    }

    cleanup_training(state, threads);
    delete n;
    free(train.data);
    free(train.labels);
    free(train.min_vals);
    free(train.max_vals);
    free(test.data);
    free(test.labels);
    free(test.min_vals);
    free(test.max_vals);
}
