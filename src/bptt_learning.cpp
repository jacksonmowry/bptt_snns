#include "backend.h"
#include "cli.h"
#include "evaluation.h"
#include "network_setup.h"
#include "network_utils.h"
#include "shared.h"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace neuro;

static void print_epoch_log(size_t epoch, size_t total_epochs,
                            const TrainingStats& stats,
                            double best_train_metric, double best_test_metric,
                            bool has_test_data, bool is_regression) {
    if (is_regression) {
        if (has_test_data) {
            printf("E%4zu/%zu  TrL: %8g  TeL: %8g  BestTeL: "
                   "%8g\n",
                   epoch + 1, total_epochs, stats.train_loss, stats.test_loss,
                   best_test_metric);
        } else {
            printf("E%4zu/%zu  TrL: %8g  BestTrL: "
                   "%8g\n",
                   epoch + 1, total_epochs, stats.train_loss,
                   best_train_metric);
        }
    } else {
        if (has_test_data) {
            printf(
                "E%4zu/%zu  TrL: %8g TrA: %7.3f  TeL: %8g TeA: %7.3f  BestTeA: "
                "%7.3f\n",
                epoch + 1, total_epochs, stats.train_loss, stats.train_acc,
                stats.test_loss, stats.test_acc, best_test_metric);
        } else {
            printf("E%4zu/%zu  TrL: %8g TrA: %7.3f  BestTrA: %7.3f\n",
                   epoch + 1, total_epochs, stats.train_loss, stats.train_acc,
                   best_train_metric);
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
    std::vector<std::string> label_strings;
    size_t train_labels;
    size_t test_labels;

    if (have_simple) {
        load_dataset(cfg.data_file.c_str(), cfg.label_file.c_str(),
                     cfg.training_percent, cfg.timeseries, &train, &test,
                     label_strings, cfg.regression);

        train_labels = test_labels = label_strings.size();
    } else {
        auto train_pair = load_dataset_single(cfg.train_data_file.c_str(),
                                              cfg.train_label_file.c_str(),
                                              cfg.timeseries, cfg.regression);
        train           = train_pair.first;
        label_strings   = std::move(train_pair.second);
        train_labels    = label_strings.size();

        auto test_pair = load_dataset_single(cfg.test_data_file.c_str(),
                                             cfg.test_label_file.c_str(),
                                             cfg.timeseries, cfg.regression);
        test           = test_pair.first;
        test_labels    = test_pair.second.size();

        // Normalize regression labels for split path (build_split_dataset
        // handles normalization for the simple path)
        if (cfg.regression) {
            if (train.labels.data && train.labels.shape) {
                compute_realdata_minmax(train.labels);
                normalize_realdata(train.labels);
            }
            if (test.labels.data && test.labels.shape) {
                compute_realdata_minmax(test.labels);
                normalize_realdata(test.labels);
            }
        }

        // Verify train and test label mappings match (same labels, same
        // order) — only for classification
        if (!cfg.regression && test.data.shape[0] > 0) {
            assert(test_pair.second == label_strings);
        }
    }

    assert(test.data.shape[0] == 0 || train_labels == test_labels);

    size_t input_neurons = train.data.shape[1] * 2;
    size_t output_neurons =
        cfg.regression ? (size_t)train.labels.shape[1] : train_labels;
    size_t hidden_neurons = cfg.hidden_neurons;
    size_t total_neurons  = input_neurons + hidden_neurons + output_neurons;

    Network* n = load_and_init_network(
        cfg.network_json_file, cfg.connectivity, cfg.learning_rate,
        cfg.decay_rate, cfg.tau, cfg.rho, cfg.timesteps, hidden_neurons,
        cfg.seed, cfg.epochs, cfg.batch_size, cfg.training_percent, cfg.threads,
        cfg.timeseries, cfg.max_delay, cfg.weight_init_stddev);

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
    size_t effective_max_delay = cfg.max_delay;
    if (n->num_nodes() == 0) {
        size_t net_max_delay = (size_t)n->get_data("proc_params")["max_delay"];
        effective_max_delay =
            cfg.max_delay < net_max_delay ? cfg.max_delay : net_max_delay;
        std::tie(neuron_count, synapse_count) = generate_network(
            n, input_neurons, hidden_neurons, output_neurons, total_neurons,
            cfg.connectivity, discrete, scale, scale_factor, min_weight,
            max_weight, max_threshold, effective_max_delay,
            cfg.weight_init_stddev);
        printf("Neurons: %zu, Synapses: %zu\n", neuron_count, synapse_count);
    } else {
        neuron_count  = n->num_nodes();
        synapse_count = n->num_edges();
        printf("Resuming training with Neurons: %zu, Synapses: %zu\n",
               neuron_count, synapse_count);
    }
    n->make_sorted_node_vector();

    build_run_metadata(n, argc, argv, cfg, &train, &test, input_neurons,
                       output_neurons, total_neurons, neuron_count,
                       synapse_count, discrete, min_potential, min_weight,
                       max_weight, max_threshold, leak_prop, scale,
                       scale_factor, effective_max_delay, cfg.regression);

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
        .loss_func          = cfg.loss_func,
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

    // Determine which metric to track for export
    bool has_test_data = test.data.shape && test.data.shape[0] > 0;

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

        // Export on improved accuracy (classification) or reduced loss
        // (regression)
        bool improved = false;
        if (cfg.regression) {
            double cur_loss =
                has_test_data ? stats.test_loss : stats.train_loss;
            double prev_loss = has_test_data ? best_test_loss : best_train_loss;
            improved         = cur_loss < prev_loss;
        } else {
            double cur_acc  = has_test_data ? stats.test_acc : stats.train_acc;
            double prev_acc = has_test_data ? best_test_acc : best_train_acc;
            improved        = cur_acc > prev_acc;
        }
        if (improved) {
            best_train_acc  = stats.train_acc;
            best_train_loss = stats.train_loss;
            best_test_acc   = stats.test_acc;
            best_test_loss  = stats.test_loss;

            // Sync weights from backend to network before exporting
            backend->update_weights(n);

            export_network(n, cfg, best_train_acc, best_train_loss,
                           best_test_acc, best_test_loss, label_strings,
                           cfg.regression);
        }

        double train_metric = cfg.regression ? best_train_loss : best_train_acc;
        double test_metric  = cfg.regression ? best_test_loss : best_test_acc;
        print_epoch_log(epoch, cfg.epochs, stats, train_metric, test_metric,
                        has_test_data, cfg.regression);
    }

    // Confusion matrix on best saved network (if flag is set, classification
    // only)
    if (cfg.confusion_matrix && !cfg.regression) {
        run_confusion_matrix(cfg, train, test, label_strings, input_neurons,
                             hidden_neurons, output_neurons, cfg.timesteps,
                             cfg.timeseries);
    }

    // Finalize: sync weights to network
    backend->update_weights(n);

    // Cleanup
    backend.reset();
    delete n;

    free_dataset(&train);
    free_dataset(&test);

    return 0;
}
