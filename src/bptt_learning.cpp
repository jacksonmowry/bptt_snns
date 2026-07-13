#include "backend.h"
#include "cli.h"
#include "data_utils.h"
#include "network_setup.h"
#include "network_utils.h"
#include "shared.h"
#include "training.h"
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace neuro;

static void print_epoch_log(size_t epoch, size_t total_epochs,
                            const TrainingStats& stats, double best_train_acc,
                            double best_test_acc, bool has_test_data) {
    if (has_test_data) {
        printf("E%4zu/%zu  TrL: %8g TrA: %7.3f  TeL: %8g TeA: %7.3f  BestTeA: "
               "%7.3f\n",
               epoch + 1, total_epochs, stats.train_loss, stats.train_acc,
               stats.test_loss, stats.test_acc, best_test_acc);
    } else {
        printf("E%4zu/%zu  TrL: %8g TrA: %7.3f  BestTrA: %7.3f\n", epoch + 1,
               total_epochs, stats.train_loss, stats.train_acc, best_train_acc);
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
    assert(test.shape[0] == 0 || train_labels == test_labels);

    /* Verify train and test label mappings match (same labels, same order) */
    if (test.shape[0] > 0) {
        for (int i = 0; i < (int)train_labels; i++) {
            if (strcmp(train.label_strings[i], test.label_strings[i])) {
                fprintf(stderr, "Mismatch between train & test labels:\n");

                fprintf(stderr, "Train: [");
                for (size_t i = 0; i < train_labels; i++) {
                    fprintf(stderr, "%s", train.label_strings[i]);

                    if (i != train_labels - 1) {
                        fprintf(stderr, " ");
                    }
                }
                fprintf(stderr, "]\n");

                fprintf(stderr, "Test: [");
                for (size_t i = 0; i < test_labels; i++) {
                    fprintf(stderr, "%s", test.label_strings[i]);

                    if (i != test_labels - 1) {
                        fprintf(stderr, " ");
                    }
                }
                fprintf(stderr, "]\n");
                exit(1);
            }
        }
    }

    size_t input_neurons =
        (cfg.timeseries) ? train.shape[1] * 2 : train.shape[1] * 2;
    size_t output_neurons = train_labels;
    size_t hidden_neurons = cfg.hidden_neurons;
    size_t total_neurons  = input_neurons + hidden_neurons + output_neurons;

    Network* n = load_and_init_network(
        cfg.network_json_file, cfg.connectivity, cfg.learning_rate,
        cfg.decay_rate, cfg.tau, cfg.rho, cfg.timesteps, hidden_neurons,
        cfg.seed, cfg.epochs, cfg.batch_size, cfg.training_percent, cfg.threads,
        cfg.timeseries);

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
            cfg.connectivity, discrete, scale, scale_factor, min_weight,
            max_weight, max_threshold);
        printf("Neurons: %zu, Synapses: %zu\n", neuron_count, synapse_count);
    } else {
        neuron_count  = n->num_nodes();
        synapse_count = n->num_edges();
        printf("Resuming training with Neurons: %zu, Synapses: %zu\n",
               neuron_count, synapse_count);
    }
    n->make_sorted_node_vector();

    build_run_metadata(n, argc, argv, cfg, input_neurons, output_neurons,
                       total_neurons, neuron_count, synapse_count, discrete,
                       min_potential, min_weight, max_weight, max_threshold,
                       leak_prop, scale, scale_factor);

    NetworkConfiguration nc = {
        .n              = n,
        .input_neurons  = input_neurons,
        .hidden_neurons = hidden_neurons,
        .output_neurons = output_neurons,
        .layer_offsets  = {0, input_neurons, input_neurons + hidden_neurons},
        .total_neurons  = total_neurons,
        .max_incoming   = 0,
        .max_outgoing   = 0,
        .timesteps      = cfg.timesteps,
        .timeseries     = cfg.timeseries,
        .min_potential  = min_potential,
        .leak           = leak,
        .scale_factor   = scale_factor,
        .steps          = scale,
        .discrete       = discrete,
        .min_weight     = min_weight,
        .max_weight     = max_weight,
        .spike_value_factor = spike_value_factor,
    };

    // Compute max_in/outgoing from network topology
    size_t max_incoming = 0;
    size_t max_outgoing = 0;
    for (size_t i = 0; i < total_neurons; i++) {
        auto* node       = n->get_node(i);
        size_t out_count = node->outgoing.size();
        if (out_count > max_outgoing) {
            max_outgoing = out_count;
        }

        size_t in_count = node->incoming.size();
        if (in_count > max_incoming) {
            max_incoming = in_count;
        }
    }
    nc.max_outgoing = max_outgoing;
    nc.max_incoming = max_incoming;

    // Create backend via factory
    auto backend = create_backend(cfg, nc, train, test);

    // Determine which accuracy metric to track for export
    bool has_test_data = test.shape[0] > 0;

    // Training loop
    // "Best" stats are updated when a new best network is found, not
    // indivdually per stat
    double best_train_acc  = 0.0;
    double best_train_loss = DBL_MAX;
    double best_test_acc   = 0.0;
    double best_test_loss  = DBL_MAX;
    for (size_t epoch = 0; epoch < cfg.epochs; ++epoch) {
        TrainingStats stats = backend->get_stats();

        backend->do_one_epoch(epoch);
        stats = backend->get_stats();

        // Export only on new high accuracy
        double cur_best_acc  = has_test_data ? stats.test_acc : stats.train_acc;
        double prev_best_acc = has_test_data ? best_test_acc : best_train_acc;
        if (cur_best_acc > prev_best_acc) {
            best_train_acc  = stats.train_acc;
            best_train_loss = stats.train_loss;
            best_test_acc   = stats.test_acc;
            best_test_loss  = stats.test_loss;

            export_network(n, cfg, best_train_acc, best_train_loss,
                           best_test_acc, best_test_loss,
                           (const char**)train.label_strings,
                           (int)train_labels);
        }

        print_epoch_log(epoch, cfg.epochs, stats, best_train_acc, best_test_acc,
                        has_test_data);
    }

    // Finalize: sync weights to network
    backend->update_weights(n);

    // Cleanup (OpenCL destructor runs final CPU eval if applicable)
    backend.reset();
    delete n;
    free(train.data);
    free(train.labels);
    free(train.min_vals);
    free(train.max_vals);
    free(train.shape);

    bool free_train = train.label_strings != test.label_strings;

    for (int i = 0; i < train.label_strings_count; i++) {
        free(train.label_strings[i]);
        train.label_strings[i] = NULL;
    }
    free(train.label_strings);
    train.label_strings = NULL;

    if (free_train) {
        for (int i = 0; i < test.label_strings_count; i++) {
            free(test.label_strings[i]);
            test.label_strings[i] = NULL;
        }
        free(test.label_strings);
    }

    free(test.data);
    free(test.labels);
    free(test.min_vals);
    free(test.max_vals);
    free(test.shape);

    return 0;
}
