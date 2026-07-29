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
#include <sstream>
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
    fprintf(stderr, "  --test                  Use test min/max instead of train "
                    "(optional, default: false)\n");
    fprintf(stderr, "  --evaluate              Use evaluate_sample() for local "
                    "prediction (optional)\n");
    fprintf(stderr, "  --output_csv <file>     Write regression predictions to CSV "
                    "(default: stdout)\n");
    fprintf(stderr, "  --csv_delimiter <char>  CSV delimiter character (default: ,)\n");
    fprintf(stderr, "  -h, --help              Show this help message\n");
    fprintf(stderr, "\n");
}

struct CliArgs {
    string network_json;
    bool test;
    bool evaluate;
    string output_csv;
    char csv_delimiter;
};

static int parse_args(int argc, char* argv[], CliArgs* out) {
    out->test           = false;
    out->evaluate       = false;
    out->output_csv     = "";
    out->csv_delimiter  = ',';

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
        } else if (arg == "--output_csv") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --output_csv requires a value\n");
                print_help(argv[0]);
                return -1;
            }
            i++;
            out->output_csv = argv[i];
        } else if (arg == "--csv_delimiter") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --csv_delimiter requires a value\n");
                print_help(argv[0]);
                return -1;
            }
            i++;
            out->csv_delimiter = argv[i][0];
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

static json read_json_array(const json& obj, const string& key) {
    if (!obj.contains(key) || !obj[key].is_array()) {
        return json::array();
    }
    return obj[key];
}

static int main_evaluate(const string& network_json_path, bool test,
                         const string& output_csv, char csv_delim) {
    // Open and parse the network JSON file
    ifstream fstream(network_json_path);
    if (!fstream.is_open()) {
        fprintf(stderr, "Error: failed to open '%s'\n", network_json_path.c_str());
        return 1;
    }

    json network_json;
    try {
        fstream >> network_json;

        const json& other = network_json.at("Associated_Data").at("other");

        // Extract metadata
        size_t timesteps  = other.value("timesteps", 0);
        size_t hidden_neurons = (size_t)other.value("hidden_neurons", 0);
        size_t output_neurons = (size_t)other.value("output_neurons", 0);
        size_t input_neurons  = (size_t)other.value("input_neurons", 0);
        bool timeseries     = other.value("timeseries", false);
        bool regression     = other.value("regression", false);

        // Select min/max based on --test flag
        const json& data_min_json =
            test ? read_json_array(other, "test_data_min")
                 : read_json_array(other, "train_data_min");
        const json& data_max_json =
            test ? read_json_array(other, "test_data_max")
                 : read_json_array(other, "train_data_max");

        if (data_min_json.empty() || data_max_json.empty()) {
            fprintf(stderr, "Error: no min/max data found in JSON\n");
            return 1;
        }

        if (timesteps == 0 || hidden_neurons == 0 || output_neurons == 0 ||
            input_neurons == 0) {
            fprintf(stderr, "Error: network JSON missing required fields "
                            "(timesteps=%zu, hidden=%zu, output=%zu, "
                            "input=%zu)\n",
                    timesteps, hidden_neurons, output_neurons, input_neurons);
            return 1;
        }

        size_t n_features = input_neurons / 2;

        vector<double> min_vals, max_vals;
        for (auto& v : data_min_json) min_vals.push_back(v.get<double>());
        for (auto& v : data_max_json) max_vals.push_back(v.get<double>());

        // For regression: load label stats from nested train_labels / test_labels
        vector<double> label_min, label_max;
        int label_dims = 1;
        int label_shape_1 = 1;
        int label_shape_2 = 1;

        if (regression) {
            /* Use train_labels if present, else test_labels. */
            const json* labels_obj = nullptr;
            if (test) {
                if (other.contains("test_labels") && other["test_labels"].is_object()) {
                    labels_obj = &other["test_labels"];
                } else if (other.contains("train_labels") && other["train_labels"].is_object()) {
                    labels_obj = &other["train_labels"];
                }
            } else {
                if (other.contains("train_labels") && other["train_labels"].is_object()) {
                    labels_obj = &other["train_labels"];
                }
            }

            if (!labels_obj || labels_obj->empty()) {
                fprintf(stderr, "Error: regression network JSON missing "
                                "label metadata (train_labels or test_labels)\n");
                return 1;
            }

            const json& labels_shape_json =
                (*labels_obj).contains("shape") ? (*labels_obj)["shape"] : json::array({1});
            const json& labels_min_json =
                (*labels_obj).contains("min") ? (*labels_obj)["min"] : json::array();
            const json& labels_max_json =
                (*labels_obj).contains("max") ? (*labels_obj)["max"] : json::array();

            if (labels_min_json.empty() || labels_max_json.empty()) {
                fprintf(stderr, "Error: regression network JSON missing "
                                "label min/max metadata\n");
                return 1;
            }

            if (!labels_shape_json.empty() && labels_shape_json.is_array()) {
                label_dims = (int)labels_shape_json.size();
                label_shape_1 = labels_shape_json[0].get<int>();
                if (label_dims >= 2) {
                    label_shape_2 = labels_shape_json[1].get<int>();
                }
            }

            for (auto& v : labels_min_json) label_min.push_back(v.get<double>());
            for (auto& v : labels_max_json) label_max.push_back(v.get<double>());
        }

        // For classification: load label mapping
        vector<string> label_strings;
        if (other.contains("label_mapping")) {
            for (auto& v : other["label_mapping"]) {
                label_strings.push_back(v.get<string>());
            }
        }

        // Load network and create processor
        auto* net = new Network();
        net->from_json(network_json);
        Processor* p = nullptr;
        load_network(&p, net);

        // Determine output source
        FILE* out_fp = stdout;
        if (!output_csv.empty()) {
            out_fp = fopen(output_csv.c_str(), "w");
            if (!out_fp) {
                fprintf(stderr, "Error: failed to open '%s' for writing\n",
                        output_csv.c_str());
                delete p;
                delete net;
                return 1;
            }
        }

        // Read batches from stdin
        size_t sample_count = 0;
        while (true) {
            vector<double> raw_values(n_features);
            bool got_all = true;
            for (size_t i = 0; i < n_features; i++) {
                if (!(cin >> raw_values[i])) {
                    got_all = false;
                    break;
                }
            }
            if (!got_all) break;

            // Normalize using data min/max
            vector<double> normalized(n_features);
            for (size_t i = 0; i < n_features; i++) {
                double range = max_vals[i] - min_vals[i];
                if (range == 0.0) {
                    // Match normalize_realdata() behavior: NaN skips spiking
                    normalized[i] = 0.0 / 0.0; // NaN
                } else {
                    normalized[i] = (raw_values[i] - min_vals[i]) / range;
                }
            }

            // Create temporary normalized dataset for encode_spikes
            Dataset tmp_ds;
            Dataset* tmp_ds_ptr = &tmp_ds;

            if (timeseries) {
                /* 3D dataset: [1, n_features, timesteps]
                 * Repeat each normalized value across all timesteps */
                int ts = (int)timesteps;
                int n_cols = (int)n_features * ts;
                double* norm_3d = new double[n_cols];
                for (size_t f = 0; f < n_features; f++) {
                    for (int t = 0; t < ts; t++) {
                        norm_3d[(size_t)f * ts + t] = normalized[f];
                    }
                }
                tmp_ds.data.data   = norm_3d;
                tmp_ds.data.shape  = (int*)malloc(3 * sizeof(int));
                tmp_ds.data.shape[0] = 1;
                tmp_ds.data.shape[1] = (int)n_features;
                tmp_ds.data.shape[2] = ts;
                tmp_ds.data.dims     = 3;
            } else {
                double* norm_data = new double[n_features];
                memcpy(norm_data, normalized.data(), n_features * sizeof(double));
                tmp_ds.data.data   = norm_data;
                tmp_ds.data.shape  = (int*)malloc(2 * sizeof(int));
                tmp_ds.data.shape[0] = 1;
                tmp_ds.data.shape[1] = (int)n_features;
                tmp_ds.data.dims     = 2;
            }
            tmp_ds.labels.data   = nullptr;
            tmp_ds.labels.dims   = 1;
            tmp_ds.labels.shape  = nullptr;
            tmp_ds.timeseries    = timeseries;

            if (regression) {
                /* Regression: evaluate and denormalize output */
                // evaluate_sample returns argmax class index which doesn't
                // apply for regression — instead we need the raw spike logits
                p->clear_activity();
                encode_spikes(p, &tmp_ds, 0, timesteps, timeseries, input_neurons);

                // Accumulate output logits across timesteps
                vector<double> output_logits(output_neurons, 0.0);
                for (size_t t = 0; t < timesteps; t++) {
                    p->run(1);
                    const vector<int>& neuron_counts = p->neuron_counts();
                    for (size_t o = 0; o < output_neurons; o++) {
                        size_t output_neuron_idx = input_neurons +
                            hidden_neurons + o;
                        output_logits[o] += neuron_counts[output_neuron_idx];
                    }
                }

                // Average over timesteps to get normalized prediction
                for (size_t o = 0; o < output_neurons; o++) {
                    output_logits[o] /= (double)timesteps;
                }

                // Denormalize using label min/max
                vector<double> denorm(output_neurons);
                if (label_dims == 1) {
                    double range = label_max[0] - label_min[0];
                    if (range == 0.0) {
                        denorm[0] = label_min[0];
                    } else {
                        denorm[0] = label_min[0] +
                                    output_logits[0] * range;
                    }
                } else {
                    for (size_t o = 0; o < output_neurons; o++) {
                        if (o < label_min.size()) {
                            double range = label_max[o] - label_min[o];
                            if (range == 0.0) {
                                denorm[o] = label_min[o];
                            } else {
                                denorm[o] = label_min[o] +
                                            output_logits[o] * range;
                            }
                        }
                    }
                }

                // Write CSV row: denormalized predictions only
                for (size_t o = 0; o < output_neurons; o++) {
                    fprintf(out_fp, "%.*g%c", (int)20, denorm[o],
                            o < output_neurons - 1 ? csv_delim : '\n');
                }

            } else {
                /* Classification: evaluate_sample returns predicted class */
                int pred = evaluate_sample(p, tmp_ds, 0, hidden_neurons,
                                           output_neurons, timesteps,
                                           timeseries, input_neurons);
                if (pred >= 0 && (size_t)pred < label_strings.size()) {
                    fprintf(out_fp, "%s\n", label_strings[pred].c_str());
                } else {
                    fprintf(out_fp, "%d\n", pred);
                }
            }

            free(tmp_ds.data.shape);
            if (tmp_ds.data.data) {
                delete[] (double*)tmp_ds.data.data;
            }
            sample_count++;
        }

        if (out_fp != stdout) {
            fclose(out_fp);
        }

        // Cleanup
        delete p;
        delete net;

    } catch (const json::exception& e) {
        fprintf(stderr, "Error: JSON access failed: %s\n", e.what());
        return 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}

static int main_client(const string& network_json_path, bool test) {
    // Open and parse the network JSON file
    ifstream fstream(network_json_path);
    if (!fstream.is_open()) {
        fprintf(stderr, "Error: failed to open '%s'\n",
                network_json_path.c_str());
        return 1;
    }

    json network_json;
    try {
        fstream >> network_json;

        const json& other = network_json.at("Associated_Data").at("other");

        size_t timesteps = other.value("timesteps", 0);

        // Select min/max based on --test flag
        const json& data_min =
            test ? other.value("test_data_min", json::array())
                 : other.value("train_data_min", json::array());
        const json& data_max =
            test ? other.value("test_data_max", json::array())
                 : other.value("train_data_max", json::array());

        if (data_min.empty() || data_max.empty()) {
            fprintf(stderr, "Error: no min/max data found in JSON\n");
            return 1;
        }

        size_t n = data_min.size();
        if (n == 0 || timesteps == 0) {
            fprintf(stderr, "Error: no min/max data or timesteps=0\n");
            return 1;
        }

        vector<double> min_vals, max_vals;
        for (auto& v : data_min) min_vals.push_back(v.get<double>());
        for (auto& v : data_max) max_vals.push_back(v.get<double>());

        // Client protocol mode: encode spikes and output commands
        vector<vector<bool>> spikes(n * 2, vector<bool>(timesteps, false));

        printf("ML %s\n", network_json_path.c_str());

        while (true) {
            vector<double> values(n);
            bool got_all = true;
            for (size_t i = 0; i < n; i++) {
                if (!(cin >> values[i])) {
                    got_all = false;
                    break;
                }
            }
            if (!got_all) break;

            // Encode spikes (normalizes internally with min/max)
            spikes = encode_spike_raster(values.data(), n, timesteps,
                                          min_vals.data(), max_vals.data(),
                                          false);

            for (size_t i = 0; i < n * 2; i++) {
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

int main(int argc, char* argv[]) {
    CliArgs args;
    int rc = parse_args(argc, argv, &args);

    if (rc != 0) {
        return 1;
    }

    if (args.network_json.empty()) {
        return 0;
    }

    if (args.evaluate) {
        return main_evaluate(args.network_json, args.test,
                             args.output_csv, args.csv_delimiter);
    } else {
        return main_client(args.network_json, args.test);
    }
}
