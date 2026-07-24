#include "data_utils.h"
#include "evaluation.h"
#include "network_utils.h"
#include "nlohmann/json.hpp"
#include "framework.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::json;
using namespace std;
using namespace neuro;

static void print_help(const char* prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --network_json <file>   Path to network JSON config "
                    "file (required)\n");
    fprintf(stderr, "  --test                  Enable test mode (optional, "
                    "default: false)\n");
    fprintf(stderr, "  --evaluate              Use evaluate_sample() for local "
                    "prediction (optional)\n");
    fprintf(stderr, "  -h, --help              Show this help message\n");
    fprintf(stderr, "\n");
}

struct CliArgs {
    string network_json;
    bool test;
    bool evaluate;  /* Use evaluate_sample() for local evaluation */
};

static int parse_args(int argc, char* argv[], CliArgs* out) {
    out->test = false;
    out->evaluate = false;

    if (argc < 2) {
        print_help(argv[0]);
        return -1;
    }

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            return 0;
        }

        if (arg == "--network_json") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --network_json requires a value\n");
                print_help(argv[0]);
                return -1;
            }
            i++;
            out->network_json = argv[i];
        } else if (arg == "--test") {
            out->test = true;
        } else if (arg == "--evaluate") {
            out->evaluate = true;
        } else {
            fprintf(stderr, "Error: unknown argument '%s'\n", arg.c_str());
            print_help(argv[0]);
            return -1;
        }
    }

    if (out->network_json.empty()) {
        fprintf(stderr, "Error: --network_json is required\n");
        print_help(argv[0]);
        return -1;
    }

    return 0;
}

int main(int argc, char* argv[]) {
    CliArgs args;
    int rc = parse_args(argc, argv, &args);

    if (rc != 0) {
        return 1;
    }

    if (args.network_json.empty()) {
        // help was printed, exit cleanly
        return 0;
    }

    // Open and parse the network JSON file
    ifstream fstream(args.network_json);
    if (!fstream.is_open()) {
        fprintf(stderr, "Error: failed to open '%s'\n",
                args.network_json.c_str());
        return 1;
    }

    json network_json;
    try {
        fstream >> network_json;

        // Fields live under Associated_Data.other
        const json& other = network_json.at("Associated_Data").at("other");

        // Extract metadata from loaded network JSON
        size_t timesteps = other.value("timesteps", 0);

        // Select min/max arrays based on --test flag
        const json& data_min =
            args.test ? other.value("test_data_min", json::array())
                      : other.value("train_data_min", json::array());
        const json& data_max =
            args.test ? other.value("test_data_max", json::array())
                      : other.value("train_data_max", json::array());

        // Convert JSON arrays to vectors for use
        vector<double> min_vals, max_vals;
        for (auto& v : data_min) {
            min_vals.push_back(v.get<double>());
        }
        for (auto& v : data_max) {
            max_vals.push_back(v.get<double>());
        }

        // Validate extracted data
        if (min_vals.size() != max_vals.size()) {
            fprintf(stderr, "Error: min/max array size mismatch (%zu vs %zu)\n",
                    min_vals.size(), max_vals.size());
            fstream.close();
            return 1;
        }
        size_t n = min_vals.size();
        if (n == 0) {
            fprintf(stderr, "Error: no min/max data found in JSON\n");
            fstream.close();
            return 1;
        }
        if (timesteps == 0) {
            fprintf(stderr, "Error: timesteps must be > 0\n");
            fstream.close();
            return 1;
        }
        size_t input_neurons = n * 2;

        if (args.evaluate) {
            /* Evaluate mode: use evaluate_sample() for local prediction */
            /* Extract timing and topology from JSON metadata */
            size_t input_neurons = (size_t)other.value("input_neurons", 0);
            size_t hidden_neurons = (size_t)other.value("hidden_neurons", 0);
            size_t output_neurons = (size_t)other.value("output_neurons", 0);
            bool timeseries = other.value("timeseries", false);

            /* Validate required fields */
            if (hidden_neurons == 0 || output_neurons == 0) {
                fprintf(stderr, "Error: network JSON missing hidden_neurons "
                                "(%zu) or output_neurons (%zu)\n",
                        hidden_neurons, output_neurons);
                return 1;
            }

            /* Load network and create processor */
            auto* net = new Network();
            net->from_json(network_json);
            Processor* p = nullptr;
            load_network(&p, net);

            /* Read batches of n floats from stdin until EOF */
            while (true) {
                vector<double> values(n);
                bool got_all = true;
                for (size_t i = 0; i < n; i++) {
                    if (!(cin >> values[i])) {
                        got_all = false;
                        break;
                    }
                }
                if (!got_all) {
                    break;
                }

                /* Create a temporary Dataset from input values */
                Dataset tmp_ds;
                tmp_ds.data = values.data();
                tmp_ds.shape = (int*)malloc(2 * sizeof(int));
                tmp_ds.shape[0] = 1;  /* one sample */
                tmp_ds.shape[1] = n;  /* n features */
                tmp_ds.min_vals = min_vals.data();
                tmp_ds.max_vals = max_vals.data();
                tmp_ds.timeseries = false;
                tmp_ds.dims = 2;
                tmp_ds.labels = nullptr;
                tmp_ds.label_strings = nullptr;
                tmp_ds.label_strings_count = 0;

                /* Evaluate sample */
                int pred = evaluate_sample(p, tmp_ds, 0, hidden_neurons,
                                           output_neurons, timesteps,
                                           timeseries, input_neurons);
                printf("PRED %d\n", pred);

                free(tmp_ds.shape);
            }

            /* Cleanup */
            delete p;
            delete net;

        } else {
            /* Client protocol mode: encode spikes and output commands */
            /* spikes[input_neuron][timestep] = true if spike fires */
            vector<vector<bool>> spikes(input_neurons,
                                        vector<bool>(timesteps, false));

            printf("ML %s\n", args.network_json.c_str());

            // Read batches of n floats from stdin until EOF
            while (true) {
                vector<double> values(n);
                bool got_all = true;
                for (size_t i = 0; i < n; i++) {
                    if (!(cin >> values[i])) {
                        got_all = false;
                        break;
                    }
                }
                if (!got_all) {
                    break;
                }

                // Reset spikes
                for (size_t i = 0; i < input_neurons; i++) {
                    fill(spikes[i].begin(), spikes[i].end(), false);
                }

                // Encode spikes (non-timeseries logic from data_utils.cpp)
                for (size_t input = 0; input < n; input++) {
                    double range = max_vals[input] - min_vals[input];
                    if (range <= 0.0) {
                        fprintf(stderr,
                                "Warning: range <= 0 for input %zu, skipping\n",
                                input);
                        continue;
                    }
                    double x = (values[input] - min_vals[input]) / range;
                    if (!std::isfinite(x)) {
                        fprintf(stderr,
                                "Warning: non-finite normalized value for input "
                                "%zu, skipping\n",
                                input);
                        continue;
                    }
                    double inv_x = 1.0 - x;

                    if (x > 0.0) {
                        double step = 1.0 / x;
                        if (step > 0.0) {
                            for (double j = 0.0; j < (double)timesteps; j += step) {
                                spikes[input * 2][(size_t)j] = true;
                            }
                        }
                    }
                    if (inv_x > 0.0) {
                        double step = 1.0 / inv_x;
                        if (step > 0.0) {
                            for (double j = 0.0; j < (double)timesteps; j += step) {
                                spikes[input * 2 + 1][(size_t)j] = true;
                            }
                        }
                    }
                }

                // Print as 1/0 with no spacing
                for (size_t i = 0; i < input_neurons; i++) {
                    printf("ASR %zu ", i);
                    for (size_t t = 0; t < timesteps; t++) {
                        cout << (spikes[i][t] ? 1 : 0);
                    }
                    cout << endl;
                }

                printf("RUN %zu\n", timesteps);
                printf("OC\n");
                printf("CA\n");
            }
        }

        fstream.close();
        return 0;

    } catch (const json::exception& e) {
        fprintf(stderr, "Error: JSON access failed: %s\n", e.what());
        fstream.close();
        return 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        fstream.close();
        return 1;
    }

    return 0;
}
