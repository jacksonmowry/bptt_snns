#include "backend.h"
#include "cli.h"
#include "data_utils.h"
#include "network_setup.h"
#include "network_utils.h"
#include "shared.h"
#include "training.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

using namespace std;
using namespace neuro;

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
        .max_outgoing   = 0,
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

    // Compute max_outgoing from network topology
    size_t max_outgoing = 0;
    for (size_t i = 0; i < total_neurons; i++) {
        auto* node = n->get_node(i);
        size_t out_count = node->outgoing.size();
        if (out_count > max_outgoing) {
            max_outgoing = out_count;
        }
    }
    nc.max_outgoing = max_outgoing;

    // Create backend via factory
    auto backend = create_backend(cfg, n, nc, train, test, state,
                                  batch_size, learning_rate, decay_rate,
                                  rho, tau);

    // Training loop
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        backend->do_one_epoch(epoch);
    }

    // Finalize: sync weights, run final eval (OpenCL), export JSON
    backend->finalize();

    // Cleanup
    backend.reset();
    free(state->batch_order);
    free(state->tas);
    free(state->tids);
    delete state;
    delete n;
    free(train.data);
    free(train.labels);
    free(train.min_vals);
    free(train.max_vals);
    free(test.data);
    free(test.labels);
    free(test.min_vals);
    free(test.max_vals);

    return 0;
}
