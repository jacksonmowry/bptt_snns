#include "nlohmann/json.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::json;
using namespace std;

static void print_help(const char* prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --network_json <file>   Path to network JSON config file (required)\n");
    fprintf(stderr, "  --test                  Enable test mode (optional, default: false)\n");
    fprintf(stderr, "  -h, --help              Show this help message\n");
    fprintf(stderr, "\n");
}

struct CliArgs {
    string network_json;
    bool test;
};

static int parse_args(int argc, char* argv[], CliArgs* out) {
    out->test = false;

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
        fprintf(stderr, "Error: failed to open '%s'\n", args.network_json.c_str());
        return 1;
    }

    json network_json;
    try {
        fstream >> network_json;
    } catch (const json::exception& e) {
        fprintf(stderr, "Error: failed to parse JSON: %s\n", e.what());
        fstream.close();
        return 1;
    }
    fstream.close();

    // Fields live under Associated_Data.other
    const json& other = network_json.at("Associated_Data").at("other");

    // Extract metadata from loaded network JSON
    size_t timesteps = other.value("timesteps", 0);

    // Select min/max arrays based on --test flag
    const json& data_min = args.test ? other.value("test_data_min", json::array())
                                     : other.value("train_data_min", json::array());
    const json& data_max = args.test ? other.value("test_data_max", json::array())
                                     : other.value("train_data_max", json::array());

    // Convert JSON arrays to vectors for use
    vector<double> min_vals, max_vals;
    for (auto& v : data_min) min_vals.push_back(v.get<double>());
    for (auto& v : data_max) max_vals.push_back(v.get<double>());

    // Skeleton: print parsed data for now
    cout << "network_json loaded, " << network_json.size() << " top-level keys" << endl;
    cout << "timesteps: " << timesteps << endl;
    cout << "min_vals size: " << min_vals.size() << endl;
    cout << "max_vals size: " << max_vals.size() << endl;
    if (args.test) {
        cout << "test mode enabled" << endl;
    }

    return 0;
}
