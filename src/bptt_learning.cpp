#include "backend.h"
#include "cli.h"
#include "data_utils.h"
#include "network_setup.h"
#include "network_utils.h"
#include "shared.h"
#include "training.h"
#include <cassert>
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
#include <algorithm>
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
        if (have_simple) {
            load_dataset_2d(cfg.data_file.c_str(), cfg.label_file.c_str(),
                            cfg.training_percent, &train, &test);
        } else {
            load_dataset_2d_single(cfg.train_data_file.c_str(),
                                   cfg.train_label_file.c_str(), &train);
            load_dataset_2d_single(cfg.test_data_file.c_str(),
                                   cfg.test_label_file.c_str(), &test);
        }
    } else {
        if (have_simple) {
            load_dataset(cfg.data_file.c_str(), cfg.label_file.c_str(),
                         cfg.training_percent, &train, &test);
        } else {
            load_dataset_single(cfg.train_data_file.c_str(),
                                cfg.train_label_file.c_str(), &train);
            load_dataset_single(cfg.test_data_file.c_str(),
                                cfg.test_label_file.c_str(), &test);
        }
    }

    size_t train_labels = label_count(&train);
    size_t test_labels  = label_count(&test);
    assert(test.shape[0] == 0 || train_labels == test_labels);

    /* Verify train and test label mappings match (same labels, same order)
     */
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
                       scale_factor, effective_max_delay);

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

            // Sync weights from backend to network before exporting
            backend->update_weights(n);

            export_network(n, cfg, best_train_acc, best_train_loss,
                           best_test_acc, best_test_loss,
                           (const char**)train.label_strings,
                           (int)train_labels);
        }

        print_epoch_log(epoch, cfg.epochs, stats, best_train_acc, best_test_acc,
                        has_test_data);
    }

    /* Confusion matrix on best saved network (if flag is set) */
    if (cfg.confusion_matrix && !cfg.network_json_out.empty()) {
        FILE* fp = fopen(cfg.network_json_out.c_str(), "r");
        if (fp) {
            fclose(fp);

            /* Load the best saved network JSON */
            json best_json;
            {
                std::ifstream ifs(cfg.network_json_out);
                ifs >> best_json;
            }
            auto* best_net = new neuro::Network();
            best_net->from_json(best_json);

            /* Create a processor for the best network */
            neuro::Processor* p = nullptr;
            load_network(&p, best_net);

            /* Choose dataset: test if present, otherwise train */
            const Dataset* eval_dataset = &test;
            if (!has_test_data) {
                eval_dataset = &train;
            }

            size_t num_samples = eval_dataset->shape[0];
            std::vector<int> true_labels(num_samples);
            std::vector<int> pred_labels(num_samples);

            for (size_t idx = 0; idx < num_samples; idx++) {
                p->clear_activity();

                /* Encode input spikes directly using processor API */
                for (size_t input = 0; input < input_neurons / 2; input++) {
                    double range = eval_dataset->max_vals[input] -
                                   eval_dataset->min_vals[input];
                    if (cfg.timeseries) {
                        size_t encoding_window = cfg.timesteps /
                                                 eval_dataset->shape[2];
                        assert(encoding_window > 0);
                        for (int column_t = 0;
                             column_t < (int)eval_dataset->shape[2];
                             column_t++) {
                            double enc_start =
                                column_t * (double)encoding_window;
                            double enc_end = enc_start + encoding_window;

                            double x     = (eval_dataset->data[(idx * eval_dataset->shape[1] *
                                         eval_dataset->shape[2]) +
                                        (input * eval_dataset->shape[2]) + column_t] -
                                            eval_dataset->min_vals[input]) /
                                           range;
                            double inv_x = 1.0 - x;

                            if (x > 0.0) {
                                for (double j = enc_start;
                                     j < enc_end;
                                     j += 1.0 / x) {
                                    p->apply_spike({(int)input * 2,
                                                    (double)(int)j, 1.0});
                                }
                            }
                            if (inv_x > 0.0) {
                                for (double j = enc_start;
                                     j < enc_end;
                                     j += 1.0 / inv_x) {
                                    p->apply_spike(
                                        {(int)input * 2 + 1,
                                         (double)(int)j, 1.0});
                                }
                            }
                        }
                    } else {
                        double x     =
                            (eval_dataset->data[idx * eval_dataset->shape[1] +
                                                 input] -
                                 eval_dataset->min_vals[input]) /
                            range;
                        double inv_x = 1.0 - x;
                        if (x > 0.0) {
                            for (double j = 0.0;
                                 j < (double)cfg.timesteps;
                                 j += 1.0 / x) {
                                p->apply_spike(
                                    {(int)input * 2,
                                     (double)(size_t)j, 1.0});
                            }
                        }
                        if (inv_x > 0.0) {
                            for (double j = 0.0;
                                 j < (double)cfg.timesteps;
                                 j += 1.0 / inv_x) {
                                p->apply_spike(
                                    {(int)input * 2 + 1,
                                     (double)(size_t)j, 1.0});
                            }
                        }
                    }
                }

                /* Run the network and accumulate output counts like training does */
                std::vector<int> output_logits(output_neurons, 0);
                for (size_t t = 0; t < cfg.timesteps; t++) {
                    p->run(1);
                    const std::vector<int>& neuron_counts = p->neuron_counts();
                    for (int o = 0; o < (int)output_neurons; o++) {
                        size_t output_neuron_idx =
                            input_neurons + hidden_neurons + o;
                        output_logits[o] +=
                            neuron_counts[output_neuron_idx];
                    }
                }
                int pred = 0;
                int max_count = output_logits[0];
                for (int o = 1; o < (int)output_neurons; o++) {
                    if (output_logits[o] > max_count) {
                        max_count = output_logits[o];
                        pred = o;
                    }
                }

                true_labels[idx] = (int)eval_dataset->labels[idx];
                pred_labels[idx] = pred;
            }

            /* Build confusion matrix */
            size_t num_classes = (size_t)train.label_strings_count;
            std::vector<std::vector<size_t>> cm(
                num_classes, std::vector<size_t>(num_classes, 0));
            size_t correct = 0;
            for (size_t idx = 0; idx < num_samples; idx++) {
                int t = true_labels[idx];
                int p_ = pred_labels[idx];
                if (t >= 0 && (size_t)t < num_classes &&
                    p_ >= 0 && (size_t)p_ < num_classes) {
                    cm[(size_t)t][(size_t)p_]++;
                    if (t == p_) {
                        correct++;
                    }
                }
            }

            /* Compute column widths: max of label string length and cell width */
            vector<size_t> col_widths(num_classes, 0);
            for (size_t j = 0; j < num_classes; j++) {
                size_t label_len = strlen(train.label_strings[j]);
                for (size_t i = 0; i < num_classes; i++) {
                    char buf[32];
                    snprintf(buf, sizeof(buf), "%zu", cm[i][j]);
                    col_widths[j] =
                        std::max(col_widths[j], std::max(label_len, strlen(buf)));
                }
            }

            /* Total width of all column data areas */
            size_t col_data_width = 0;
            for (size_t j = 0; j < num_classes; j++) {
                col_data_width += col_widths[j] + 2; /* space + content + space */
            }
            col_data_width += num_classes; /* "|" separators */

            /* Left label column width */
            size_t left_label_width = 0;
            for (size_t i = 0; i < num_classes; i++) {
                left_label_width =
                    std::max(left_label_width, strlen(train.label_strings[i]));
            }
            left_label_width =
                std::max(left_label_width, (size_t)strlen("True"));
            left_label_width += 1; /* 1 space padding */

            /* Build separator line to match data column widths */
            /* Data column = 1 space + col_widths[j] content + 1 pipe = col_widths[j] + 2 chars */
            string sep = string(left_label_width, '-') + "|";
            for (size_t j = 0; j < num_classes; j++) {
                sep += string(col_widths[j] + 1, '-') + "|";
            }

            printf("Confusion Matrix:\n");

            /* Axis labels: "Predicted" centered above columns */
            printf(" %*s", (int)left_label_width - 1, "");
            printf("|");
            printf(" %*s", (int)(col_data_width / 2 - 4), "Predicted");
            printf("\n");

            /* Column labels + "True" axis label on the left */
            printf(" %*s", (int)left_label_width - 1, "True");
            printf("|");
            for (size_t j = 0; j < num_classes; j++) {
                /* Left-align column label to match left edge of data cell */
                printf(" ");
                printf("%s", train.label_strings[j]);
                for (size_t k = strlen(train.label_strings[j]); k < col_widths[j]; k++)
                    printf(" ");
                printf("|");
            }
            printf("\n");
            printf("%s\n", sep.c_str());

            /* Data rows */
            for (size_t i = 0; i < num_classes; i++) {
                printf(" %*s", (int)left_label_width - 1, train.label_strings[i]);
                printf("|");
                for (size_t j = 0; j < num_classes; j++) {
                    char buf[32];
                    snprintf(buf, sizeof(buf), "%zu", cm[i][j]);
                    printf(" %*s", (int)col_widths[j], buf);
                    printf("|");
                }
                printf("\n");
            }

            double accuracy = (double)correct / (double)num_samples;
            printf("Accuracy: %.3f (%zu/%zu)\n", accuracy, correct, num_samples);

            /* Cleanup */
            delete p;
            delete best_net;
        } else {
            fprintf(
                stderr,
                "Warning: --confusion_matrix set but best network file not "
                "found at: %s\n",
                cfg.network_json_out.c_str());
        }
    }

    // Finalize: sync weights to network
    backend->update_weights(n);

    // Cleanup
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
