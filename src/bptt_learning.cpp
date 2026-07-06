// Best Parameters:
//   learning_rate: 0.00911106696288836
//   decay_rate: 2.4817122209750068e-05
//   tau: 0.8163744927232162
//   rho: 0.5486399999186418
#include "csv.h"
#include "framework.hpp"
#include "shared.h"
#include "math_utils.h"
#include "data_utils.h"
#include "network_utils.h"
#include "forward_backward.h"
#include "optimizer.h"
#include "threading.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cfloat>
#include <cstddef>
#include <cstdlib>
#include <cstdarg>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <pthread.h>
#include <string>
#include <unordered_set>
#include <vector>
#include <ctime>
#include <sys/time.h>
#include <sys/time.h>

using namespace std;
using namespace neuro;
using nlohmann::json;

#define NUM_LAYERS (3)

struct CliConfig {
    std::string network_json_file;
    std::string data_file;
    std::string label_file;
    std::string train_data_file;
    std::string train_label_file;
    std::string test_data_file;
    std::string test_label_file;
    bool timeseries = false;
    double connectivity = 0.2;
    double learning_rate = 0.008;
    double decay_rate = 0.0001;
    double tau = 0.95;
    double rho = 1.4;
    size_t timesteps = 32;
    size_t hidden_neurons = 16;
    unsigned long seed = (unsigned long)time(NULL);
    size_t epochs = 10;
    size_t batch_size = 1;
    double training_percent = 0.8;
    std::string network_json_out;
    size_t threads = 1;
    bool show_help = false;
};

static int cli_error(const char* fmt, ...) {
    va_list ap;
    fprintf(stderr, "Error: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    return 1;
}

static int check_range(double v, double lo, double hi,
                       const char* name) {
    if (v < lo || v > hi) {
        return cli_error("--%s must be in [%.1f, %.1f]", name, lo, hi);
    }
    return 0;
}

static int check_pos(double v, const char* name) {
    if (v <= 0.0) {
        return cli_error("--%s must be > 0", name);
    }
    return 0;
}

static int parse_double_arg(int& i, int argc, char* argv[],
                            double* out, const char* name) {
    if (++i >= argc) {
        return cli_error("--%s requires a value", name);
    }
    char* end = nullptr;
    errno = 0;
    double v = strtod(argv[i], &end);
    if (end == argv[i] || *end != '\0' || errno != 0) {
        return cli_error("--%s: invalid numeric value '%s'", name, argv[i]);
    }
    *out = v;
    return 0;
}

static int parse_ulong_arg(int& i, int argc, char* argv[],
                           unsigned long* out, const char* name) {
    if (++i >= argc) {
        return cli_error("--%s requires a value", name);
    }
    char* end = nullptr;
    errno = 0;
    unsigned long v = strtoul(argv[i], &end, 0);
    if (end == argv[i] || *end != '\0' || errno != 0) {
        return cli_error("--%s: invalid integer value '%s'", name, argv[i]);
    }
    *out = v;
    return 0;
}

static int parse_cli(int argc, char* argv[], CliConfig* cfg) {
    int i = 1;

    while (i < argc) {
        string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            cfg->show_help = true;
            ++i;
            continue;
        }
        else if (arg == "--network_json" || arg == "-n") {
            if (++i >= argc) return cli_error("--network_json requires a value");
            cfg->network_json_file = argv[i];
        }
        else if (arg == "--data_file" || arg == "-d") {
            if (++i >= argc) return cli_error("--data_file requires a value");
            cfg->data_file = argv[i];
        }
        else if (arg == "--label_file" || arg == "-l") {
            if (++i >= argc) return cli_error("--label_file requires a value");
            cfg->label_file = argv[i];
        }
        else if (arg == "--train_data_file" || arg == "-a") {
            if (++i >= argc) return cli_error("--train_data_file requires a value");
            cfg->train_data_file = argv[i];
        }
        else if (arg == "--train_label_file" || arg == "-i") {
            if (++i >= argc) return cli_error("--train_label_file requires a value");
            cfg->train_label_file = argv[i];
        }
        else if (arg == "--test_data_file" || arg == "-j") {
            if (++i >= argc) return cli_error("--test_data_file requires a value");
            cfg->test_data_file = argv[i];
        }
        else if (arg == "--test_label_file" || arg == "-k") {
            if (++i >= argc) return cli_error("--test_label_file requires a value");
            cfg->test_label_file = argv[i];
        }
        else if (arg == "--timeseries" || arg == "-b") {
            cfg->timeseries = true;
        }
        else if (arg == "--connectivity" || arg == "-c" || arg == "-S") {
            double v;
            int rc = parse_double_arg(i, argc, argv, &v, "connectivity");
            if (rc) return rc;
            rc = check_range(v, 0.0, 1.0, "connectivity");
            if (rc) return rc;
            cfg->connectivity = v;
        }
        else if (arg == "--learning_rate" || arg == "-r") {
            double v;
            int rc = parse_double_arg(i, argc, argv, &v, "learning_rate");
            if (rc) return rc;
            rc = check_range(v, 0.0, 1.0, "learning_rate");
            if (rc) return rc;
            cfg->learning_rate = v;
        }
        else if (arg == "--decay_rate" || arg == "-e") {
            double v;
            int rc = parse_double_arg(i, argc, argv, &v, "decay_rate");
            if (rc) return rc;
            rc = check_range(v, 0.0, 1.0, "decay_rate");
            if (rc) return rc;
            cfg->decay_rate = v;
        }
        else if (arg == "--tau" || arg == "-u") {
            double v;
            int rc = parse_double_arg(i, argc, argv, &v, "tau");
            if (rc) return rc;
            rc = check_pos(v, "tau");
            if (rc) return rc;
            cfg->tau = v;
        }
        else if (arg == "--rho" || arg == "-o") {
            double v;
            int rc = parse_double_arg(i, argc, argv, &v, "rho");
            if (rc) return rc;
            rc = check_pos(v, "rho");
            if (rc) return rc;
            cfg->rho = v;
        }
        else if (arg == "--timesteps" || arg == "-t") {
            unsigned long v;
            int rc = parse_ulong_arg(i, argc, argv, &v, "timesteps");
            if (rc) return rc;
            if (v == 0) return cli_error("--timesteps must be > 0");
            cfg->timesteps = v;
        }
        else if (arg == "--hidden_neurons" || arg == "-H") {
            unsigned long v;
            int rc = parse_ulong_arg(i, argc, argv, &v, "hidden_neurons");
            if (rc) return rc;
            if (v == 0) return cli_error("--hidden_neurons must be > 0");
            cfg->hidden_neurons = v;
        }
        else if (arg == "--seed" || arg == "-s") {
            unsigned long v;
            int rc = parse_ulong_arg(i, argc, argv, &v, "seed");
            if (rc) return rc;
            cfg->seed = v;
        }
        else if (arg == "--epochs" || arg == "-p") {
            unsigned long v;
            int rc = parse_ulong_arg(i, argc, argv, &v, "epochs");
            if (rc) return rc;
            cfg->epochs = v;
        }
        else if (arg == "--batch_size" || arg == "-B") {
            unsigned long v;
            int rc = parse_ulong_arg(i, argc, argv, &v, "batch_size");
            if (rc) return rc;
            cfg->batch_size = v;
        }
        else if (arg == "--training_percent" || arg == "-P") {
            double v;
            int rc = parse_double_arg(i, argc, argv, &v, "training_percent");
            if (rc) return rc;
            rc = check_range(v, 0.0, 1.0, "training_percent");
            if (rc) return rc;
            cfg->training_percent = v;
        }
        else if (arg == "--network_json_out" || arg == "-N") {
            if (++i >= argc) return cli_error("--network_json_out requires a value");
            cfg->network_json_out = argv[i];
        }
        else if (arg == "--threads" || arg == "-T") {
            unsigned long v;
            int rc = parse_ulong_arg(i, argc, argv, &v, "threads");
            if (rc) return rc;
            if (v == 0) return cli_error("--threads must be > 0");
            cfg->threads = v;
        }
        else {
            return cli_error("Unknown argument: %s", arg.c_str());
        }

        ++i;
    }

    return 0;
}

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [OPTIONS]...\n\n", prog);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -n, --network_json         FILE    Network JSON path\n");
    fprintf(stderr, "  -d, --data_file            FILE    Data file path\n");
    fprintf(stderr, "  -l, --label_file           FILE    Label file path\n");
    fprintf(stderr, "  -a, --train_data_file      FILE    Train data file path\n");
    fprintf(stderr, "  -i, --train_label_file     FILE    Train label file path\n");
    fprintf(stderr, "  -j, --test_data_file       FILE    Test data file path\n");
    fprintf(stderr, "  -k, --test_label_file      FILE    Test label file path\n");
    fprintf(stderr, "\nEither (-d + -l) OR (-a + -i + -j + -k) required.\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -b, --timeseries                    Enable timeseries mode\n");
    fprintf(stderr, "  -S, --connectivity     FLOAT        Neuron connection chance (0,1]\n");
    fprintf(stderr, "  -r, --learning_rate    FLOAT        Learning rate (0,1]\n");
    fprintf(stderr, "  -e, --decay_rate       FLOAT        Decay rate (0,1]\n");
    fprintf(stderr, "  -u, --tau              FLOAT        Tau (>0)\n");
    fprintf(stderr, "  -o, --rho              FLOAT        Rho (>0)\n");
    fprintf(stderr, "  -t, --timesteps        UINT         Timestep count\n");
    fprintf(stderr, "  -H, --hidden_neurons   UINT         Hidden layer size\n");
    fprintf(stderr, "  -s, --seed             UINT         Random seed\n");
    fprintf(stderr, "  -p, --epochs           UINT         Training epochs\n");
    fprintf(stderr, "  -B, --batch_size       UINT         Batch size\n");
    fprintf(stderr, "  -P, --training_percent FLOAT        Train split ratio (0,1]\n");
    fprintf(stderr, "  -N, --network_json_out FILE         Output network JSON\n");
    fprintf(stderr, "  -T, --threads          UINT         Thread count\n");
    fprintf(stderr, "  -h, --help                          Show this help\n");
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
        cli_error("--network_json is required");
        print_usage(argv[0]);
        return 1;
    }

    bool have_simple = !cfg.data_file.empty() && !cfg.label_file.empty();
    bool have_split = !cfg.train_data_file.empty() &&
                      !cfg.train_label_file.empty() &&
                      !cfg.test_data_file.empty() &&
                      !cfg.test_label_file.empty();

    if (have_simple && have_split) {
        cli_error("cannot specify both (-d + -l) and (-a + -i + -j + -k); choose one");
        print_usage(argv[0]);
        return 1;
    }

    if (!have_simple && !have_split) {
        cli_error("either (-d + -l) OR (-a + -i + -j + -k) are required");
        print_usage(argv[0]);
        return 1;
    }

    char* network_json_file = const_cast<char*>(cfg.network_json_file.c_str());
    char* data_file = const_cast<char*>(cfg.data_file.c_str());
    char* label_file = const_cast<char*>(cfg.label_file.c_str());
    char* train_data_file = const_cast<char*>(cfg.train_data_file.c_str());
    char* train_label_file = const_cast<char*>(cfg.train_label_file.c_str());
    char* test_data_file = const_cast<char*>(cfg.test_data_file.c_str());
    char* test_label_file = const_cast<char*>(cfg.test_label_file.c_str());
    bool timeseries = cfg.timeseries;
    double connectivity = cfg.connectivity;
    double learning_rate = cfg.learning_rate;
    double decay_rate = cfg.decay_rate;
    double tau = cfg.tau;
    double rho = cfg.rho;
    size_t timesteps = cfg.timesteps;
    size_t hidden_neurons = cfg.hidden_neurons;
    unsigned long seed = cfg.seed;
    size_t epochs = cfg.epochs;
    size_t batch_size = cfg.batch_size;
    double training_percent = cfg.training_percent;
    char* network_json_out = const_cast<char*>(cfg.network_json_out.c_str());
    size_t threads = cfg.threads;

    srand(seed);
    srand48(seed);

    Dataset train;
    Dataset test;

    if (timeseries) {
        assert(false);
        load_dataset_2d(data_file, label_file, training_percent, &train, &test);
    } else {
        if (data_file && label_file) {
            load_dataset(data_file, label_file, training_percent, &train,
                         &test);
        } else {
            load_dataset_single(train_data_file, train_label_file, &train);
            load_dataset_single(test_data_file, test_label_file, &test);
        }
    }

    size_t train_labels = label_count(&train);
    size_t test_labels  = label_count(&test);
    assert(test.observations == 0 || train_labels == test_labels);

    size_t input_neurons =
        (timeseries) ? train.rows_per_observation * 2 : train.cols * 2;
    size_t output_neurons = train_labels;
    size_t total_neurons  = input_neurons + hidden_neurons + output_neurons;

    json emptynet;
    ifstream fin(network_json_file);
    fin >> emptynet;
    fin.close();

    Network* n = new Network();
    n->from_json(emptynet);

    // Auto-detect: if network has nodes/edges, try to load metadata and
    // override all CLI params.  Only generate from scratch when the network
    // is truly empty (no nodes or edges).
    if (n->num_nodes() > 0 && n->num_edges() > 0) {
        bool has_other = false;
        for (size_t ki = 0; ki < n->data_keys().size(); ki++) {
            if (n->data_keys()[ki] == "other") {
                has_other = true;
                break;
            }
        }

        if (has_other) {
            json other = n->get_data("other");
            printf("Warning: Loading network from %s. CLI arguments will be "
                   "overridden by saved metadata:\n",
                   network_json_file);

            // Helper lambda: read a double from metadata and override CLI value
            auto override_double = [&](const std::string& key,
                                       double& target,
                                       const char* param_name) {
                if (other.count(key)) {
                    target = other[key].get<double>();
                    printf("[metadata override] %s: %.10g (from saved network)\n",
                           param_name, target);
                }
            };

            // Helper lambda: read a size_t from metadata and override CLI value
            auto override_size = [&](const std::string& key,
                                     size_t& target,
                                     const char* param_name) {
                if (other.count(key)) {
                    target = other[key].get<size_t>();
                    printf("[metadata override] %s: %zu (from saved network)\n",
                           param_name, target);
                }
            };

            // Helper lambda: read a bool from metadata and override CLI value
            auto override_bool = [&](const std::string& key,
                                     bool& target,
                                     const char* param_name) {
                if (other.count(key)) {
                    target = other[key].get<bool>();
                    printf("[metadata override] %s: %s (from saved network)\n",
                           param_name, target ? "true" : "false");
                }
            };

            // Helper lambda: read a string from metadata and override CLI value
            auto override_string = [&](const std::string& key,
                                       char*& target,
                                       const char* param_name) {
                if (other.count(key)) {
                    std::string val = other[key].get<std::string>();
                    if (target != NULL) free(target);
                    target = strdup(val.c_str());
                    printf("[metadata override] %s: %s (from saved network)\n",
                           param_name, target);
                }
            };

            override_double("connectivity", connectivity,
                            "--connectivity");
            override_double("learning_rate", learning_rate,
                            "--learning_rate");
            override_double("decay_rate", decay_rate,
                            "--decay_rate");
            override_double("tau", tau, "--tau");
            override_double("rho", rho, "--rho");
            override_size("timesteps", timesteps, "--timesteps");
            override_size("hidden_neurons", hidden_neurons,
                          "--hidden_neurons");
            override_size("seed", seed, "--seed");
            override_size("epochs", epochs, "--epochs");
            override_size("batch_size", batch_size, "--batch_size");
            override_double("training_percent", training_percent,
                            "--training_percent");
            override_size("threads", threads, "--threads");
            override_bool("timeseries", timeseries, "--timeseries");
            override_string("data_file", data_file, "--data_file");
            override_string("label_file", label_file, "--label_file");
            override_string("train_data_file", train_data_file,
                            "--train_data_file");
            override_string("train_label_file", train_label_file,
                            "--train_label_file");
            override_string("test_data_file", test_data_file,
                            "--test_data_file");
            override_string("test_label_file", test_label_file,
                            "--test_label_file");
            override_string("network_json_out", network_json_out,
                            "--network_json_out");

            printf("Loaded metadata from %s\n", network_json_file);
        } else {
            fprintf(stderr,
                    "Warning: Network has nodes/edges but no metadata found. "
                    "Using default CLI parameters.\n");
        }
    }

    bool discrete        = n->get_data("proc_params")["discrete"];
    string leak_prop     = n->get_data("proc_params")["leak_mode"];
    bool leak            = leak_prop == "all";
    double min_potential = n->get_data("proc_params")["min_potential"];
    double min_weight    = n->get_data("proc_params")["min_weight"];
    double max_weight    = n->get_data("proc_params")["max_weight"];
    double max_threshold = n->get_data("proc_params")["max_threshold"];
    int scale            = 0;

    if (discrete) {
        // Symmetric scale around zero, rounded up to next bit
        scale = max(abs(min_weight), abs(max_weight)) * 2 + 1;
        scale = pow(2.0, ceil(log2(scale)));
    }

    double scale_factor = 2.0 / scale;

    if (discrete) {
        min_potential *= scale_factor;
    }

    const size_t layer_sizes[3] = {input_neurons, hidden_neurons,
                                   output_neurons};

    NetworkConfiguration nc = {
        .n = n,

        .input_neurons  = input_neurons,
        .hidden_neurons = hidden_neurons,
        .output_neurons = output_neurons,
        .layer_offsets  = {0, input_neurons, input_neurons + hidden_neurons},
        .total_neurons  = total_neurons,

        .timesteps  = timesteps,
        .timeseries = timeseries,

        .min_potential = min_potential,
        .leak          = leak,
        .scale_factor  = scale_factor,
        .steps         = scale,
        .discrete      = discrete,
        .min_weight    = min_weight,
        .max_weight    = max_weight,
    };

    size_t neuron_count  = 0;
    size_t synapse_count = 0;

    if (n->num_nodes() == 0) {
        // Generate nodes/edges if the network is empty
        for (size_t i = 0; i < NUM_LAYERS; i++) {
            for (size_t j = 0; j < layer_sizes[i]; j++) {
                n->add_node(neuron_count)->set("Threshold", max_threshold);

                if (i == 0) {
                    n->add_input(neuron_count);
                } else if (i == NUM_LAYERS - 1) {
                    n->add_output(neuron_count);
                }

                neuron_count++;
            }
        }

        for (size_t i = 0; i < total_neurons; i++) {
            for (size_t j = 0; j < total_neurons; j++) {
                if (drand48() < (1.0 - connectivity)) {
                    continue;
                }

                Edge* e = n->add_edge(i, j);
                synapse_count++;

                double weight = normal(0.0, 0.1);
                if (discrete) {
                    weight = quantize(weight, scale, min_weight, max_weight) /
                             scale_factor;
                }
                int delay = rand() % 7 + 1;

                e->set(n->get_edge_property("Weight")->index, weight);
                e->set(n->get_edge_property("Delay")->index, delay);
            }
        }

        printf("Neurons: %zu, Synapses: %zu\n", neuron_count, synapse_count);
    } else {
        // Use existing network structure
        neuron_count  = n->num_nodes();
        synapse_count = n->num_edges();
        printf("Resuming training with Neurons: %zu, Synapses: %zu\n",
               neuron_count, synapse_count);
    }
    n->make_sorted_node_vector();

    // --- Capture run metadata for full reproducibility ---
    // CLI arguments
    json cli_args = json::array();
    for (int i = 0; i < argc; i++) {
        cli_args.push_back(std::string(argv[i]));
    }

    // Git commit hash
    std::string git_commit;
    {
        FILE* fp = popen("git rev-parse HEAD 2>/dev/null", "r");
        if (fp) {
            char buf[64] = {};
            if (fgets(buf, sizeof(buf), fp)) {
                git_commit = buf;
                if (!git_commit.empty() && git_commit.back() == '\n')
                    git_commit.pop_back();
            }
            pclose(fp);
        }
    }

    // Compile timestamp
    auto compile_time = std::time(nullptr);

    // Start time
    struct timeval tv;
    gettimeofday(&tv, nullptr);

    // Build metadata for Associated_Data -> other
    json run_metadata = json::object();
    run_metadata["cli_args"] = cli_args;
    run_metadata["git_commit"] = git_commit.empty() ? "unknown" : git_commit;
    run_metadata["compile_time"] = compile_time;
    run_metadata["start_time"] = (double)tv.tv_sec + tv.tv_usec / 1000000.0;
    run_metadata["seed"] = seed;
    run_metadata["connectivity"] = connectivity;
    run_metadata["learning_rate"] = learning_rate;
    run_metadata["decay_rate"] = decay_rate;
    run_metadata["tau"] = tau;
    run_metadata["rho"] = rho;
    run_metadata["timesteps"] = timesteps;
    run_metadata["hidden_neurons"] = hidden_neurons;
    run_metadata["epochs"] = epochs;
    run_metadata["batch_size"] = batch_size;
    run_metadata["training_percent"] = training_percent;
    run_metadata["threads"] = threads;
    run_metadata["timeseries"] = timeseries;
    run_metadata["network_json"] = network_json_file ? std::string(network_json_file) : "";
    run_metadata["data_file"] = data_file ? std::string(data_file) : "";
    run_metadata["label_file"] = label_file ? std::string(label_file) : "";
    run_metadata["train_data_file"] = train_data_file ? std::string(train_data_file) : "";
    run_metadata["train_label_file"] = train_label_file ? std::string(train_label_file) : "";
    run_metadata["test_data_file"] = test_data_file ? std::string(test_data_file) : "";
    run_metadata["test_label_file"] = test_label_file ? std::string(test_label_file) : "";
    run_metadata["network_json_out"] = network_json_out ? std::string(network_json_out) : "";
    run_metadata["input_neurons"] = input_neurons;
    run_metadata["output_neurons"] = output_neurons;
    run_metadata["total_neurons"] = total_neurons;
    run_metadata["neuron_count"] = neuron_count;
    run_metadata["synapse_count"] = synapse_count;
    run_metadata["discrete"] = discrete;
    run_metadata["min_potential"] = min_potential;
    run_metadata["min_weight"] = min_weight;
    run_metadata["max_weight"] = max_weight;
    run_metadata["max_threshold"] = max_threshold;
    run_metadata["leak_mode"] = leak_prop;
    run_metadata["scale"] = scale;
    run_metadata["scale_factor"] = discrete ? scale_factor : 1.0;

    // Merge with existing Associated_Data -> other if any
    json existing_other = json::object();
    bool has_other = false;
    for (size_t ki = 0; ki < n->data_keys().size(); ki++) {
        if (n->data_keys()[ki] == "other") { has_other = true; break; }
    }
    if (has_other) {
        existing_other = n->get_data("other");
    }
    // Existing keys take priority (e.g. proc_name), new metadata fills rest
    for (auto it = run_metadata.begin(); it != run_metadata.end(); ++it) {
        if (existing_other.find(it.key()) == existing_other.end()) {
            existing_other[it.key()] = it.value();
        }
    }
    n->set_data("other", existing_other);

    vector<vector<double>> weights(total_neurons);
    vector<vector<int>> delays(total_neurons);
    vector<double> thresholds(total_neurons);
    vector<vector<double>> m_weights(total_neurons);
    vector<vector<double>> v_weights(total_neurons);
    vector<vector<double>> delta_W(total_neurons);
    double b1_t = 1.0;
    double b2_t = 1.0;

    for (size_t i = 0; i < total_neurons; i++) {
        thresholds[i] = n->get_node(i)->get("Threshold") *
                        ((discrete) ? scale_factor : 1.0);
        weights[i].reserve(n->get_node(i)->incoming.size());
        delays[i].reserve(n->get_node(i)->incoming.size());

        m_weights[i].resize(n->get_node(i)->incoming.size());
        v_weights[i].resize(n->get_node(i)->incoming.size());
        delta_W[i].resize(n->get_node(i)->incoming.size());

        for (size_t j = 0; j < n->get_node(i)->incoming.size(); j++) {
            Edge* e = n->get_node(i)->incoming[j];

            weights[i].push_back(e->get("Weight") *
                                 ((discrete) ? scale_factor : 1.0));
            delays[i].push_back(e->get("Delay"));
        }
    }

    ThreadArgs* tas          = (ThreadArgs*)calloc(threads, sizeof(*tas));
    pthread_t* tids          = (pthread_t*)calloc(threads, sizeof(*tids));
    int max_idx              = -1;
    pthread_mutex_t mut      = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t have_work = PTHREAD_COND_INITIALIZER;
    pthread_cond_t done_work = PTHREAD_COND_INITIALIZER;
    bool train_p             = true;
    bool die                 = false;
    size_t* batch_order =
        (size_t*)calloc(train.observations, sizeof(*batch_order));
    int work_idx   = 0;
    int done_count = 0;

    for (size_t i = 0; i < threads; i++) {
        tas[i] = ThreadArgs(total_neurons, timesteps, output_neurons, rho, tau,
                            &weights, &delays, &thresholds, &nc, batch_order,
                            &train, &test, &max_idx, &work_idx, &done_count,
                            &mut, &have_work, &done_work, &train_p, &die);

        for (size_t neuron = 0; neuron < total_neurons; neuron++) {
            tas[i].tb.delta_W[neuron].resize(
                n->get_node(neuron)->incoming.size());
        }
    }

    for (size_t i = 0; i < threads; i++) {
        pthread_create(tids + i, NULL, worker, (void*)(tas + i));
    }

    puts("Beginning training");
    double best_train_loss = DBL_MAX;
    double best_test_loss  = DBL_MAX;
    double best_train_acc  = 0.0;
    double best_test_acc   = 0.0;
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss = 0.0;
        size_t correct    = 0;

        // Reset work index before each epoch
        pthread_mutex_lock(&mut);
        work_idx = 0;
        train_p  = true;

        // Shuffle the batch order for randomness
        for (int i = 0; i < train.observations; i++) {
            batch_order[i] = i;
        }
        for (int i = 0; i < train.observations; i++) {
            size_t j       = rand() % train.observations;
            size_t tmp     = batch_order[i];
            batch_order[i] = batch_order[j];
            batch_order[j] = tmp;
        }
        pthread_mutex_unlock(&mut);

        // Batch processing loop
        for (int batch_start = 0; batch_start < train.observations;
             batch_start += batch_size) {
            size_t current_batch_size = min(
                (size_t)batch_size, train.observations - (size_t)batch_start);

            pthread_mutex_lock(&mut);
            work_idx   = batch_start;
            done_count = 0;
            max_idx    = batch_start + current_batch_size;
            pthread_cond_broadcast(&have_work);
            pthread_mutex_unlock(&mut);

            pthread_mutex_lock(&mut);
            while (done_count < (int)current_batch_size) {
                pthread_cond_wait(&done_work, &mut);
            }
            pthread_mutex_unlock(&mut);

            for (size_t i = 0; i < threads; i++) {
                epoch_loss += tas[i].loss;
                correct += tas[i].correct;
                tas[i].loss      = 0;
                tas[i].correct   = 0;
                tas[i].processed = 0;

                for (size_t row = 0; row < total_neurons; row++) {
                    for (size_t incoming = 0;
                         incoming < tas[i].tb.delta_W[row].size(); incoming++) {
                        delta_W[row][incoming] +=
                            tas[i].tb.delta_W[row][incoming];
                        tas[i].tb.delta_W[row][incoming] = 0.0;
                    }
                }
            }

            // Average gradients over the actual batch size.
            weight_updates(&nc, &train, current_batch_size, batch_size,
                           batch_start, epoch, b1_t, b2_t, m_weights, v_weights,
                           learning_rate, decay_rate, weights, delta_W);
        }

        // Training Metrics
        if (epoch_loss / (double)train.observations < best_train_loss) {
            best_train_loss = epoch_loss / (double)train.observations;
        }
        if (correct / (double)train.observations > best_train_acc) {
            best_train_acc = correct / (double)train.observations;
        }

        // Test
        double test_correct = 0.0;
        double test_loss    = 0.0;

        pthread_mutex_lock(&mut);
        work_idx   = 0;
        done_count = 0;
        max_idx    = test.observations;
        train_p    = false;
        pthread_cond_broadcast(&have_work);
        pthread_mutex_unlock(&mut);

        pthread_mutex_lock(&mut);
        while (done_count < test.observations) {
            pthread_cond_wait(&done_work, &mut);
        }
        pthread_mutex_unlock(&mut);

        pthread_mutex_lock(&mut);
        for (size_t i = 0; i < threads; i++) {
            test_loss += tas[i].loss;
            test_correct += tas[i].correct;
            tas[i].loss      = 0;
            tas[i].correct   = 0;
            tas[i].processed = 0;
        }
        max_idx = -1;
        pthread_mutex_unlock(&mut);

        if (test.observations > 0) {
            test_correct /= test.observations;
            test_loss /= test.observations;

            if (test_correct > best_test_acc) {
                best_test_acc = test_correct;
            }
            if (test_loss < best_test_loss) {
                best_test_loss = test_loss;
            }
        }

        printf(
            "Epoch: %4zu/%zu, Loss: %10g (Best: %10g), Acc: %10g (Best: %10g), "
            "TestLoss: %10g (Best: %10g), TestAcc: %10g (Best: %10g)\n",
            epoch + 1, epochs, epoch_loss / (double)train.observations,
            best_train_loss, correct / (double)train.observations,
            best_train_acc, test_loss, best_test_loss, test_correct,
            best_test_acc);

        if (!cfg.network_json_out.empty()) {
            // Update runtime metrics in metadata
            json meta = n->get_data("other");
            meta["best_train_loss"] = best_train_loss;
            meta["best_test_loss"] = best_test_loss;
            meta["best_train_acc"] = best_train_acc;
            meta["best_test_acc"] = best_test_acc;
            meta["epoch"] = epoch + 1;
            n->set_data("other", meta);

            json j;
            n->to_json(j);
            ofstream fout(network_json_out);
            if (!fout) {
                fprintf(stderr,
                        "Failed to open networks/trained.json for writing\n");
                exit(1);
            }

            fout << j << endl;
            fout.close();
        }
        // printf("%g\n", best_test_loss);
    }

    pthread_mutex_lock(&mut);
    die = true;
    pthread_mutex_unlock(&mut);
    pthread_cond_broadcast(&have_work);

    for (size_t i = 0; i < threads; i++) {
        pthread_join(tids[i], NULL);
    }

    delete n;
    free(train.data);
    free(train.labels);
    free(train.min_vals);
    free(train.max_vals);
    free(test.data);
    free(test.labels);
    free(test.min_vals);
    free(test.max_vals);
    free(batch_order);
    free(tas);
    free(tids);
}
