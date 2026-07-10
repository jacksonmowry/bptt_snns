#include "nlohmann/json.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

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

    if (rc == 0 && args.network_json.empty()) {
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
    fstream >> network_json;
    fstream.close()

    // Skeleton: print parsed data for now
    cout << "network_json loaded, " << network_json.size() << " top-level keys" << endl;
    if (args.test) {
        cout << "test mode enabled" << endl;
    }

    return 0;
}
