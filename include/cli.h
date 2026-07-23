#pragma once

#include <cstddef>
#include <string>

struct CliConfig {
    std::string network_json_file;
    std::string data_file;
    std::string label_file;
    std::string train_data_file;
    std::string train_label_file;
    std::string test_data_file;
    std::string test_label_file;
    bool timeseries         = false;
    double connectivity     = 0.2;
    double learning_rate    = 0.008;
    double decay_rate       = 0.0001;
    double tau              = 0.95;
    double rho              = 1.4;
    size_t timesteps        = 32;
    size_t hidden_neurons   = 16;
    unsigned long seed      = 0;
    size_t epochs           = 10;
    size_t batch_size       = 1;
    double training_percent = 0.8;
    std::string network_json_out;
    size_t threads      = 1;
    bool show_help      = false;
#ifdef OPENCL
    bool opencl         = false;
    size_t cpu_eval_interval =
        0; // every N epochs, read GPU weights & eval on CPU
#endif
    size_t max_delay          = 7;
    double weight_init_stddev = 0.1;
};

int parse_cli(int argc, char* argv[], CliConfig* cfg);
void print_usage(const char* prog);
