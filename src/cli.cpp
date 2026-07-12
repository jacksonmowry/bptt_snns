#include "cli.h"
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <unistd.h>

static int cli_error(const char* fmt, ...) {
    va_list ap;
    fprintf(stderr, "Error: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    return 1;
}

static int check_range(double v, double lo, double hi, const char* name) {
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

static int parse_double_arg(int& i, int argc, char* argv[], double* out,
                            const char* name) {
    if (++i >= argc) {
        return cli_error("--%s requires a value", name);
    }
    char* end = nullptr;
    errno     = 0;
    double v  = strtod(argv[i], &end);
    if (end == argv[i] || *end != '\0' || errno != 0) {
        return cli_error("--%s: invalid numeric value '%s'", name, argv[i]);
    }
    *out = v;
    return 0;
}

static int parse_ulong_arg(int& i, int argc, char* argv[], unsigned long* out,
                           const char* name) {
    if (++i >= argc) {
        return cli_error("--%s requires a value", name);
    }
    char* end       = nullptr;
    errno           = 0;
    unsigned long v = strtoul(argv[i], &end, 0);
    if (end == argv[i] || *end != '\0' || errno != 0) {
        return cli_error("--%s: invalid integer value '%s'", name, argv[i]);
    }
    *out = v;
    return 0;
}

int parse_cli(int argc, char* argv[], CliConfig* cfg) {
    int i     = 1;
    cfg->seed = (unsigned long)time(NULL);

    while (i < argc) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            cfg->show_help = true;
            ++i;
        } else if (arg == "--network_json" || arg == "-n") {
            if (++i >= argc) {
                return cli_error("--network_json requires a value");
            }
            cfg->network_json_file = argv[i];

            if (access(cfg->network_json_file.c_str(), F_OK)) {
                return cli_error("Failed to open --network_json \"%s\"",
                                 cfg->network_json_file.c_str());
            }
        } else if (arg == "--data_file" || arg == "-d") {
            if (++i >= argc) {
                return cli_error("--data_file requires a value");
            }
            cfg->data_file = argv[i];

            if (access(cfg->data_file.c_str(), F_OK)) {
                return cli_error("Failed to open --data_file \"%s\"",
                                 cfg->data_file.c_str());
            }
        } else if (arg == "--label_file" || arg == "-l") {
            if (++i >= argc) {
                return cli_error("--label_file requires a value");
            }
            cfg->label_file = argv[i];

            if (access(cfg->label_file.c_str(), F_OK)) {
                return cli_error("Failed to open --label_file \"%s\"",
                                 cfg->label_file.c_str());
            }
        } else if (arg == "--train_data_file" || arg == "-a") {
            if (++i >= argc) {
                return cli_error("--train_data_file requires a value");
            }
            cfg->train_data_file = argv[i];

            if (access(cfg->train_data_file.c_str(), F_OK)) {
                return cli_error("Failed to open --train_data_file \"%s\"",
                                 cfg->train_data_file.c_str());
            }
        } else if (arg == "--train_label_file" || arg == "-i") {
            if (++i >= argc) {
                return cli_error("--train_label_file requires a value");
            }
            cfg->train_label_file = argv[i];

            if (access(cfg->train_label_file.c_str(), F_OK)) {
                return cli_error("Failed to open --train_label_file \"%s\"",
                                 cfg->train_label_file.c_str());
            }
        } else if (arg == "--test_data_file" || arg == "-j") {
            if (++i >= argc) {
                return cli_error("--test_data_file requires a value");
            }
            cfg->test_data_file = argv[i];

            if (access(cfg->test_data_file.c_str(), F_OK)) {
                return cli_error("Failed to open --test_data_file \"%s\"",
                                 cfg->test_data_file.c_str());
            }
        } else if (arg == "--test_label_file" || arg == "-k") {
            if (++i >= argc) {
                return cli_error("--test_label_file requires a value");
            }
            cfg->test_label_file = argv[i];

            if (access(cfg->test_label_file.c_str(), F_OK)) {
                return cli_error("Failed to open --test_label_file \"%s\"",
                                 cfg->test_label_file.c_str());
            }
        } else if (arg == "--timeseries" || arg == "-b") {
            cfg->timeseries = true;
        } else if (arg == "--connectivity" || arg == "-c" || arg == "-S") {
            double v;
            int rc = parse_double_arg(i, argc, argv, &v, "connectivity");
            if (rc) {
                return rc;
            }
            rc = check_range(v, 0.0, 1.0, "connectivity");
            if (rc) {
                return rc;
            }
            cfg->connectivity = v;
        } else if (arg == "--learning_rate" || arg == "-r") {
            double v;
            int rc = parse_double_arg(i, argc, argv, &v, "learning_rate");
            if (rc) {
                return rc;
            }
            rc = check_range(v, 0.0, 1.0, "learning_rate");
            if (rc) {
                return rc;
            }
            cfg->learning_rate = v;
        } else if (arg == "--decay_rate" || arg == "-e") {
            double v;
            int rc = parse_double_arg(i, argc, argv, &v, "decay_rate");
            if (rc) {
                return rc;
            }
            rc = check_range(v, 0.0, 1.0, "decay_rate");
            if (rc) {
                return rc;
            }
            cfg->decay_rate = v;
        } else if (arg == "--tau" || arg == "-u") {
            double v;
            int rc = parse_double_arg(i, argc, argv, &v, "tau");
            if (rc) {
                return rc;
            }
            rc = check_pos(v, "tau");
            if (rc) {
                return rc;
            }
            cfg->tau = v;
        } else if (arg == "--rho" || arg == "-o") {
            double v;
            int rc = parse_double_arg(i, argc, argv, &v, "rho");
            if (rc) {
                return rc;
            }
            rc = check_pos(v, "rho");
            if (rc) {
                return rc;
            }
            cfg->rho = v;
        } else if (arg == "--timesteps" || arg == "-t") {
            unsigned long v;
            int rc = parse_ulong_arg(i, argc, argv, &v, "timesteps");
            if (rc) {
                return rc;
            }
            if (v == 0) {
                return cli_error("--timesteps must be > 0");
            }
            cfg->timesteps = v;
        } else if (arg == "--hidden_neurons" || arg == "-H") {
            unsigned long v;
            int rc = parse_ulong_arg(i, argc, argv, &v, "hidden_neurons");
            if (rc) {
                return rc;
            }
            if (v == 0) {
                return cli_error("--hidden_neurons must be > 0");
            }
            cfg->hidden_neurons = v;
        } else if (arg == "--seed" || arg == "-s") {
            unsigned long v;
            int rc = parse_ulong_arg(i, argc, argv, &v, "seed");
            if (rc) {
                return rc;
            }
            cfg->seed = v;
        } else if (arg == "--epochs" || arg == "-p") {
            unsigned long v;
            int rc = parse_ulong_arg(i, argc, argv, &v, "epochs");
            if (rc) {
                return rc;
            }
            cfg->epochs = v;
        } else if (arg == "--batch_size" || arg == "-B") {
            unsigned long v;
            int rc = parse_ulong_arg(i, argc, argv, &v, "batch_size");
            if (rc) {
                return rc;
            }
            cfg->batch_size = v;
        } else if (arg == "--training_percent" || arg == "-P") {
            double v;
            int rc = parse_double_arg(i, argc, argv, &v, "training_percent");
            if (rc) {
                return rc;
            }
            rc = check_range(v, 0.0, 1.0, "training_percent");
            if (rc) {
                return rc;
            }
            cfg->training_percent = v;
        } else if (arg == "--network_json_out" || arg == "-N") {
            if (++i >= argc) {
                return cli_error("--network_json_out requires a value");
            }
            cfg->network_json_out = argv[i];

            {
                std::string path = cfg->network_json_out;
                std::ofstream test(path, std::ios::out | std::ios::trunc);
                if (!test) {
                    return cli_error(
                        "Failed to open --network_json_out \"%s\" for writing",
                        path.c_str());
                }
            }
        } else if (arg == "--threads" || arg == "-T") {
            unsigned long v;
            int rc = parse_ulong_arg(i, argc, argv, &v, "threads");
            if (rc) {
                return rc;
            }
            if (v == 0) {
                return cli_error("--threads must be > 0");
            }
            cfg->threads = v;
        } else if (arg == "--opencl") {
            if (++i >= argc) {
                return cli_error("--opencl requires a value (true/false)");
            }
            cfg->opencl = (std::string(argv[i]) == "true");
        } else if (arg == "--opencl_timings") {
            if (++i >= argc) {
                return cli_error(
                    "--opencl_timings requires a value (true/false)");
            }
            cfg->opencl_timings = (std::string(argv[i]) == "true");
        } else if (arg == "--cpu_eval_interval") {
            unsigned long v;
            int rc = parse_ulong_arg(i, argc, argv, &v, "cpu_eval_interval");
            if (rc) {
                return rc;
            }
            cfg->cpu_eval_interval = v;
        } else {
            return cli_error("Unknown argument: %s", arg.c_str());
        }
        ++i;
    }
    return 0;
}

void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [OPTIONS]...\n\n", prog);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -n, --network_json         FILE    Network JSON path\n");
    fprintf(stderr, "  -d, --data_file            FILE    Data file path\n");
    fprintf(stderr, "  -l, --label_file           FILE    Label file path\n");
    fprintf(stderr,
            "  -a, --train_data_file      FILE    Train data file path\n");
    fprintf(stderr,
            "  -i, --train_label_file     FILE    Train label file path\n");
    fprintf(stderr,
            "  -j, --test_data_file       FILE    Test data file path\n");
    fprintf(stderr,
            "  -k, --test_label_file      FILE    Test label file path\n");
    fprintf(stderr, "\nEither (-d + -l) OR (-a + -i + -j + -k) required.\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr,
            "  -b, --timeseries                    Enable timeseries mode\n");
    fprintf(stderr, "  -S, --connectivity     FLOAT        Neuron connection "
                    "chance (0,1]\n");
    fprintf(stderr,
            "  -r, --learning_rate    FLOAT        Learning rate (0,1]\n");
    fprintf(stderr, "  -e, --decay_rate       FLOAT        Decay rate (0,1]\n");
    fprintf(stderr, "  -u, --tau              FLOAT        Tau (>0)\n");
    fprintf(stderr, "  -o, --rho              FLOAT        Rho (>0)\n");
    fprintf(stderr, "  -t, --timesteps        UINT         Timestep count\n");
    fprintf(stderr,
            "  -H, --hidden_neurons   UINT         Hidden layer size\n");
    fprintf(stderr, "  -s, --seed             UINT         Random seed\n");
    fprintf(stderr, "  -p, --epochs           UINT         Training epochs\n");
    fprintf(stderr, "  -B, --batch_size       UINT         Batch size\n");
    fprintf(stderr,
            "  -P, --training_percent FLOAT        Train split ratio (0,1]\n");
    fprintf(stderr,
            "  -N, --network_json_out FILE         Output network JSON\n");
    fprintf(stderr, "  -T, --threads          UINT         Thread count\n");
    fprintf(
        stderr,
        " N/A, --opencl                        Enable OpenCL Acceleration\n");
    fprintf(stderr, " N/A, --cpu_eval_interval UINT      CPU test every N "
                    "epochs (0=off)\n");
    fprintf(
        stderr,
        " N/A, --opencl_timings          Enable OpenCL kernel timing report\n");
    fprintf(stderr, "  -h, --help                          Show this help\n");
}
