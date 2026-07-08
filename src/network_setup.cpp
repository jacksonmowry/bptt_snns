#include "network_setup.h"
#include "math_utils.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <sys/time.h>

using nlohmann::json;

neuro::Network*
load_and_init_network(const std::string& json_file, double& connectivity,
                      double& learning_rate, double& decay_rate, double& tau,
                      double& rho, size_t& timesteps, size_t& hidden_neurons,
                      unsigned long& seed, size_t& epochs, size_t& batch_size,
                      double& training_percent, size_t& threads,
                      bool& timeseries) {
    json emptynet;
    std::ifstream fin(json_file);
    fin >> emptynet;
    fin.close();

    neuro::Network* n = new neuro::Network();
    n->from_json(emptynet);

    // Auto-detect: if network has nodes/edges, try to load metadata and
    // override all CLI params.
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
                   json_file.c_str());

            auto override_double = [&](const std::string& key, double& target,
                                       const char* param_name) {
                if (other.count(key)) {
                    target = other[key].get<double>();
                    printf(
                        "[metadata override] %s: %.10g (from saved network)\n",
                        param_name, target);
                }
            };
            auto override_size = [&](const std::string& key, size_t& target,
                                     const char* param_name) {
                if (other.count(key)) {
                    target = other[key].get<size_t>();
                    printf("[metadata override] %s: %zu (from saved network)\n",
                           param_name, target);
                }
            };
            auto override_bool = [&](const std::string& key, bool& target,
                                     const char* param_name) {
                if (other.count(key)) {
                    target = other[key].get<bool>();
                    printf("[metadata override] %s: %s (from saved network)\n",
                           param_name, target ? "true" : "false");
                }
            };

            override_double("connectivity", connectivity, "--connectivity");
            override_double("learning_rate", learning_rate, "--learning_rate");
            override_double("decay_rate", decay_rate, "--decay_rate");
            override_double("tau", tau, "--tau");
            override_double("rho", rho, "--rho");
            override_size("timesteps", timesteps, "--timesteps");
            override_size("hidden_neurons", hidden_neurons, "--hidden_neurons");
            override_size("seed", seed, "--seed");
            override_size("epochs", epochs, "--epochs");
            override_size("batch_size", batch_size, "--batch_size");
            override_double("training_percent", training_percent,
                            "--training_percent");
            override_size("threads", threads, "--threads");
            override_bool("timeseries", timeseries, "--timeseries");
            printf("Loaded metadata from %s\n", json_file.c_str());
        } else {
            fprintf(stderr,
                    "Warning: Network has nodes/edges but no metadata found. "
                    "Using default CLI parameters.\n");
        }
    }

    return n;
}

void build_run_metadata(
    neuro::Network* n, int argc, char* argv[], const CliConfig& cfg,
    size_t input_neurons, size_t output_neurons, size_t total_neurons,
    size_t neuron_count, size_t synapse_count, bool discrete,
    double min_potential, double min_weight, double max_weight,
    double max_threshold, const std::string& leak_prop, int scale,
    double scale_factor, double connectivity, double learning_rate,
    double decay_rate, double tau, double rho, size_t timesteps,
    size_t hidden_neurons, unsigned long seed, size_t epochs, size_t batch_size,
    double training_percent, size_t threads, bool timeseries) {
    // CLI arguments
    json cli_args = json::array();
    for (int i = 1; i < argc; i++) {
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
                if (!git_commit.empty() && git_commit.back() == '\n') {
                    git_commit.pop_back();
                }
            }
            pclose(fp);
        }
    }

    auto compile_time = std::time(nullptr);
    struct timeval tv;
    gettimeofday(&tv, nullptr);

    json run_metadata             = json::object();
    run_metadata["cli_args"]      = cli_args;
    run_metadata["git_commit"]    = git_commit.empty() ? "unknown" : git_commit;
    run_metadata["compile_time"]  = compile_time;
    run_metadata["start_time"]    = (double)tv.tv_sec + tv.tv_usec / 1000000.0;
    run_metadata["seed"]          = seed;
    run_metadata["connectivity"]  = connectivity;
    run_metadata["learning_rate"] = learning_rate;
    run_metadata["decay_rate"]    = decay_rate;
    run_metadata["tau"]           = tau;
    run_metadata["rho"]           = rho;
    run_metadata["timesteps"]     = timesteps;
    run_metadata["hidden_neurons"]   = hidden_neurons;
    run_metadata["epochs"]           = epochs;
    run_metadata["batch_size"]       = batch_size;
    run_metadata["training_percent"] = training_percent;
    run_metadata["threads"]          = threads;
    run_metadata["timeseries"]       = timeseries;
    run_metadata["network_json"]     = cfg.network_json_file;
    run_metadata["data_file"]        = cfg.data_file;
    run_metadata["label_file"]       = cfg.label_file;
    run_metadata["train_data_file"]  = cfg.train_data_file;
    run_metadata["train_label_file"] = cfg.train_label_file;
    run_metadata["test_data_file"]   = cfg.test_data_file;
    run_metadata["test_label_file"]  = cfg.test_label_file;
    run_metadata["network_json_out"] = cfg.network_json_out;
    run_metadata["input_neurons"]    = input_neurons;
    run_metadata["output_neurons"]   = output_neurons;
    run_metadata["total_neurons"]    = total_neurons;
    run_metadata["neuron_count"]     = neuron_count;
    run_metadata["synapse_count"]    = synapse_count;
    run_metadata["discrete"]         = discrete;
    run_metadata["min_potential"]    = min_potential;
    run_metadata["min_weight"]       = min_weight;
    run_metadata["max_weight"]       = max_weight;
    run_metadata["max_threshold"]    = max_threshold;
    run_metadata["leak_mode"]        = leak_prop;
    run_metadata["scale"]            = scale;
    run_metadata["scale_factor"]     = discrete ? scale_factor : 1.0;

    // Merge with existing Associated_Data -> other if any
    json existing_other = json::object();
    bool has_other      = false;
    for (size_t ki = 0; ki < n->data_keys().size(); ki++) {
        if (n->data_keys()[ki] == "other") {
            has_other = true;
            break;
        }
    }
    if (has_other) {
        existing_other = n->get_data("other");
    }
    for (auto it = run_metadata.begin(); it != run_metadata.end(); ++it) {
        if (existing_other.find(it.key()) == existing_other.end()) {
            existing_other[it.key()] = it.value();
        }
    }
    n->set_data("other", existing_other);
}

std::pair<size_t, size_t>
generate_network(neuro::Network* n, size_t input_neurons, size_t hidden_neurons,
                 size_t output_neurons, size_t total_neurons,
                 double connectivity, bool discrete, int scale,
                 double scale_factor, double min_weight, double max_weight,
                 double max_threshold) {
    const size_t layer_sizes[3] = {input_neurons, hidden_neurons,
                                   output_neurons};
    size_t neuron_count         = 0;
    size_t synapse_count        = 0;

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < layer_sizes[i]; j++) {
            n->add_node(neuron_count)->set("Threshold", max_threshold);

            if (i == 0) {
                n->add_input(neuron_count);
            } else if (i == 2) {
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

            neuro::Edge* e = n->add_edge(i, j);
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

    return {neuron_count, synapse_count};
}

void init_network_weights(neuro::Network* n, size_t total_neurons,
                          bool discrete, double scale_factor,
                          std::vector<std::vector<double>>& weights,
                          std::vector<std::vector<int>>& delays,
                          std::vector<double>& thresholds) {
    weights.resize(total_neurons);
    delays.resize(total_neurons);
    thresholds.resize(total_neurons);

    for (size_t i = 0; i < total_neurons; i++) {
        thresholds[i] = n->get_node(i)->get("Threshold") *
                        ((discrete) ? scale_factor : 1.0);
        weights[i].reserve(n->get_node(i)->incoming.size());
        delays[i].reserve(n->get_node(i)->incoming.size());

        for (size_t j = 0; j < n->get_node(i)->incoming.size(); j++) {
            neuro::Edge* e = n->get_node(i)->incoming[j];
            weights[i].push_back(e->get("Weight") *
                                 ((discrete) ? scale_factor : 1.0));
            delays[i].push_back(e->get("Delay"));
        }
    }
}
