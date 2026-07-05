// Best Parameters:
//   learning_rate: 0.00911106696288836
//   decay_rate: 2.4817122209750068e-05
//   tau: 0.8163744927232162
//   rho: 0.5486399999186418
#include "backtrace.h"
#include "backward_pass.h"
#include "bptt_types.h"
#include "csv.h"
#include "data_utils.h"
#include "forward_pass.h"
#include "framework.hpp"
#include "math_utils.h"
#include "opencl_utils.h"
#include "weight_updates.h"
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstddef>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <pthread.h>
#include <stddef.h>

using namespace std;
using namespace neuro;
using nlohmann::json;

// Adam/Learning Parameters
#define BETA1 (0.9)
#define BETA2 (0.999)
#define ADAM_EPS (1.0e-8)
#define NUM_LAYERS (3)

void load_network(Processor** pp, Network* net) {
    json proc_params;
    string proc_name;
    Processor* p;

    p = *pp;
    if (p == nullptr) {
        proc_params = net->get_data("proc_params");
        proc_name   = net->get_data("other")["proc_name"];
        p           = Processor::make(proc_name, proc_params);
        *pp         = p;
    }

    if (p->get_network_properties().as_json() !=
        net->get_properties().as_json()) {
        fprintf(stderr,
                "%s: load_network: Network and processor properties do not "
                "match.\n",
                __FILE__);
        exit(1);
    }

    if (!p->load_network(net)) {
        fprintf(stderr, "%s: load_network: Failed to load network.\n",
                __FILE__);
        exit(1);
    }
}

void* worker(void* arg) {
    if (!arg) {
        fprintf(stderr, "Thread spawned with NULL arg\n");
        exit(1);
    }

    ThreadArgs* ta         = (ThreadArgs*)arg;
    Processor* p           = NULL;
    bool do_backwards_pass = false;
    int my_max;

    while (true) {
        pthread_mutex_lock(ta->mut);
        if (*ta->die) {
            pthread_mutex_unlock(ta->mut);
            pthread_exit(NULL);
        }

        while (*ta->work_idx >= *ta->max_idx) {
            pthread_cond_wait(ta->have_work, ta->mut);
            if (*ta->die) {
                pthread_mutex_unlock(ta->mut);
                pthread_exit(NULL);
            }
        }
        int my_work_idx   = *ta->work_idx;
        *ta->work_idx     = my_work_idx + 1;
        my_max            = *ta->max_idx;
        do_backwards_pass = *ta->train_p;
        pthread_mutex_unlock(ta->mut);

        load_network(&p, ta->nc->n);

        while (my_work_idx < my_max) {
            EvaluationResults er = forward(
                &ta->tb, p, *ta->train_p ? ta->train : ta->test,
                *ta->train_p ? ta->order[my_work_idx] : my_work_idx, ta->nc);
            ta->loss += er.loss;
            ta->correct += er.correct;
            ta->processed++;

            if (do_backwards_pass) {
                backward(&ta->tb, ta->nc);
            }

            pthread_mutex_lock(ta->mut);
            my_work_idx   = *ta->work_idx;
            *ta->work_idx = my_work_idx + 1;
            assert(*ta->max_idx == my_max);
            do_backwards_pass = *ta->train_p;
            pthread_mutex_unlock(ta->mut);
        }

        pthread_mutex_lock(ta->mut);
        *ta->done = *ta->done + ta->processed;
        pthread_cond_signal(ta->done_work);
        pthread_mutex_unlock(ta->mut);
        delete p;
        p = NULL;
    }

    return NULL;
}

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [OPTIONS]...\n", prog);
    fprintf(stderr, "  -n, --network_json     FILE    Network JSON path\n");
    fprintf(stderr, "  -d, --data_file        FILE    Data file path\n");
    fprintf(stderr, "  -l, --label_file       FILE    Label file path\n");
    fprintf(stderr, "  -l, --train_data_file  FILE    Train data file path\n");
    fprintf(stderr, "  -l, --train_label_file FILE    Train label file path\n");
    fprintf(stderr, "  -l, --test_data_file   FILE    Test data file path\n");
    fprintf(stderr, "  -l, --test_label_file  FILE    Test label file path\n");
    fprintf(stderr,
            "  -b, --timeseries               Enable timeseries mode\n");
    fprintf(stderr, "  -S, --connectivity     FLOAT   Chance each neuron is "
                    "connected to another (0,1]\n");
    fprintf(stderr, "  -r, --learning_rate    FLOAT   Learning rate (0,1]\n");
    fprintf(stderr, "  -e, --decay_rate       FLOAT   Decay rate (0,1]\n");
    fprintf(stderr, "  -u, --tau              FLOAT   Tau (>0)\n");
    fprintf(stderr, "  -o, --rho              FLOAT   Rho (>0)\n");
    fprintf(stderr, "  -t, --timesteps        UINT    Timestep count\n");
    fprintf(stderr, "  -H, --hidden_neurons   UINT    Hidden layer size\n");
    fprintf(stderr, "  -s, --seed             UINT    Random seed\n");
    fprintf(stderr, "  -p, --epochs           UINT    Training epochs\n");
    fprintf(stderr, "  -B, --batch_size       UINT    Training batch size\n");
    fprintf(
        stderr,
        "  -p, --training_percent FLOAT   Training percent of total data\n");
    fprintf(stderr,
            "  -N, --network_json_out FILE    Network json out filename\n");
    fprintf(stderr, "  -h, --help                     Show this help\n");
}

int main(int argc, char* argv[]) {
    char* network_json_file = NULL;
    char* data_file         = NULL;
    char* label_file        = NULL;
    char* train_data_file   = NULL;
    char* train_label_file  = NULL;
    char* test_data_file    = NULL;
    char* test_label_file   = NULL;
    bool timeseries         = false;
    double connectivity     = 0.2;
    double learning_rate    = 0.008;
    double decay_rate       = 0.0001;
    double tau              = 0.95;
    double rho              = 1.4;
    size_t timesteps        = 32;
    size_t hidden_neurons   = 16;
    unsigned long seed      = (unsigned long)time(NULL);
    size_t epochs           = 10;
    size_t batch_size       = 1;
    double training_percent = 0.8;
    char* network_json_out  = NULL;
    size_t threads          = 1;

    static struct option long_options[] = {
        {"network_json", required_argument, 0, 'n'},
        {"data_file", required_argument, 0, 'd'},
        {"label_file", required_argument, 0, 'l'},
        {"train_data_file", required_argument, 0, 'a'},
        {"train_label_file", required_argument, 0, 'i'},
        {"test_data_file", required_argument, 0, 'j'},
        {"test_label_file", required_argument, 0, 'k'},
        {"timeseries", no_argument, 0, 'b'},
        {"connectivity", required_argument, 0, 'c'},
        {"learning_rate", required_argument, 0, 'r'},
        {"decay_rate", required_argument, 0, 'e'},
        {"tau", required_argument, 0, 'u'},
        {"rho", required_argument, 0, 'o'},
        {"timesteps", required_argument, 0, 't'},
        {"hidden_neurons", required_argument, 0, 'H'},
        {"seed", required_argument, 0, 's'},
        {"epochs", required_argument, 0, 'p'},
        {"batch_size", required_argument, 0, 'B'},
        {"training_percent", required_argument, 0, 'P'},
        {"network_json_out", required_argument, 0, 'N'},
        {"threads", required_argument, 0, 'T'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0},
    };

    int c;
    char* endptr;

    while ((c = getopt_long(argc, argv,
                            "n:d:l:a:i:j:k:b:c:r:e:u:o:t:H:s:hp:B:P:N:T:",
                            long_options, NULL)) != -1) {
        switch (c) {
        case 'n':
            network_json_file = optarg;
            break;
        case 'd':
            data_file = optarg;
            break;
        case 'l':
            label_file = optarg;
            break;
        case 'a':
            train_data_file = optarg;
            break;
        case 'i':
            train_label_file = optarg;
            break;
        case 'j':
            test_data_file = optarg;
            break;
        case 'k':
            test_label_file = optarg;
            break;
        case 'b':
            timeseries = true;
            break;
        case 'c':
            connectivity = strtod(optarg, &endptr);
            if (*endptr != '\0' || connectivity <= 0.0 || connectivity > 1.0) {
                fprintf(stderr,
                        "Error: Invalid or out-of-range --connectivity\n");
                return 1;
            }
            break;
        case 'r':
            learning_rate = strtod(optarg, &endptr);
            if (*endptr != '\0' || learning_rate <= 0.0 ||
                learning_rate > 1.0) {
                fprintf(stderr,
                        "Error: Invalid or out-of-range --learning_rate\n");
                return 1;
            }
            break;
        case 'e':
            decay_rate = strtod(optarg, &endptr);
            if (*endptr != '\0' || decay_rate <= 0.0 || decay_rate > 1.0) {
                fprintf(stderr,
                        "Error: Invalid or out-of-range --decay_rate\n");
                return 1;
            }
            break;
        case 'u':
            tau = strtod(optarg, &endptr);
            if (*endptr != '\0' || tau <= 0.0) {
                fprintf(stderr, "Error: Invalid or out-of-range --tau\n");
                return 1;
            }
            break;
        case 'o':
            rho = strtod(optarg, &endptr);
            if (*endptr != '\0' || rho <= 0.0) {
                fprintf(stderr, "Error: Invalid or out-of-range --rho\n");
                return 1;
            }
            break;
        case 't':
            timesteps = strtoull(optarg, &endptr, 0);
            if (*endptr != '\0' || timesteps == 0) {
                fprintf(stderr, "Error: Invalid or out-of-range --timesteps\n");
                return 1;
            }
            break;
        case 'H':
            hidden_neurons = strtoull(optarg, &endptr, 0);
            if (*endptr != '\0' || hidden_neurons == 0) {
                fprintf(stderr,
                        "Error: Invalid or out-of-range --hidden_neurons\n");
                return 1;
            }
            break;
        case 's':
            seed = strtoull(optarg, &endptr, 0);
            if (*endptr != '\0') {
                fprintf(stderr, "Error: Invalid --seed\n");
                return 1;
            }
            break;
        case 'p':
            epochs = strtoull(optarg, &endptr, 0);
            if (*endptr != '\0') {
                fprintf(stderr, "Error: Invalid --epochs\n");
                return 1;
            }
            break;
        case 'B':
            batch_size = strtoull(optarg, &endptr, 0);
            if (*endptr != '\0') {
                fprintf(stderr, "Error: Invalid --batch_size\n");
                return 1;
            }
            break;
        case 'P':
            training_percent = strtod(optarg, &endptr);
            if (*endptr != '\0' || training_percent <= 0.0 ||
                training_percent > 1.0) {
                fprintf(stderr,
                        "Error: Invalid or out-of-range --training_percent\n");
                return 1;
            }
            break;
        case 'N':
            network_json_out = optarg;
            break;
        case 'T':
            threads = strtoull(optarg, &endptr, 0);
            if (*endptr != '\0') {
                fprintf(stderr, "Error: Invalid --threads\n");
                return 1;
            }
            break;
        case 'h':
            print_usage(argv[0]);
            return 0;
        case '?':
            print_usage(argv[0]);
            return 1;
        }
    }

    // Can only have data/label files OR train/test data/label files
    // This also includes the training_percentage parameter
    if (network_json_file == NULL) {
        fprintf(stderr, "Error: --network_json is required.\n");
        print_usage(argv[0]);
        return 1;
    }

    if (!(data_file && label_file) && !(train_data_file && train_label_file &&
                                        test_data_file && test_label_file)) {
        fprintf(stderr,
                "Error: either (--data_file AND --label_file) OR "
                "(--train_data_file AND --train_label_file AND "
                "--test_data_file AND --test_label_file) are required.\n");
        print_usage(argv[0]);
        return 1;
    }

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
    if (!n) {
        fprintf(stderr, "%s:%s:%d: Unable to create network.\n", __FILE__,
                __FUNCTION__, __LINE__);
        exit(1);
    }

    bool discrete        = n->get_data("proc_params")["discrete"];
    string leak_prop     = n->get_data("proc_params")["leak_mode"];
    bool leak            = leak_prop == "all";
    double min_potential = n->get_data("proc_params")["min_potential"];
    double min_weight    = n->get_data("proc_params")["min_weight"];
    double max_weight    = n->get_data("proc_params")["max_weight"];
    double max_threshold = n->get_data("proc_params")["max_threshold"];
    double spike_value_factor =
        n->get_data("proc_params")["spike_value_factor"];
    int scale = 0;

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
        .discrete      = discrete,
    };

    size_t neuron_count  = 0;
    size_t synapse_count = 0;

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

    uint16_t max_incoming = 0;
    for (size_t i = 0; i < total_neurons; i++) {
        uint16_t incoming = 0;
        for (size_t j = 0; j < total_neurons; j++) {
            if (drand48() < (1.0 - connectivity)) {
                continue;
            }

            Edge* e = n->add_edge(i, j);
            synapse_count++;
            incoming++;

            double weight = normal(0.0, 0.1);
            if (discrete) {
                weight = quantize(weight, scale, min_weight, max_weight) /
                         scale_factor;
            }
            int delay = rand() % 7 + 1;

            e->set(n->get_edge_property("Weight")->index, weight);
            e->set(n->get_edge_property("Delay")->index, delay);
        }

        if (incoming > max_incoming) {
            max_incoming = incoming;
        }
    }

    printf("Neurons: %zu, Synapses: %zu\n", neuron_count, synapse_count);
    n->make_sorted_node_vector();

    // Encode data for opencl kernels
    double* train_encoded = (double*)malloc(train.observations * train.cols *
                                            2 * sizeof(*train_encoded));
    for (int i = 0; i < train.observations; i++) {
        for (int j = 0; j < train.cols; j++) {
            double x = (train.data[i * train.cols + j] - train.min_vals[j]) /
                       (train.max_vals[j] - train.min_vals[j]);
            double inv_x = 1.0 - x;

            if (x > 0.0) {
                train_encoded[(i * train.cols * 2) + (j * 2)] = 1.0 / x;
            }

            if (inv_x > 0.0) {
                train_encoded[(i * train.cols * 2) + (j * 2 + 1)] = 1.0 / inv_x;
            }
        }
    }

    for (int i = 0; i < 8; i++) {
        printf("%g ", train_encoded[i]);
    }
    puts("");
    puts("");

    // OpenCL host memory allocations
    int16_t* cl_thresh = (int16_t*)calloc(total_neurons, sizeof(*cl_thresh));
    int16_t* cl_weights =
        (int16_t*)calloc(total_neurons * max_incoming, sizeof(*cl_weights));
    uint16_t* cl_delays =
        (uint16_t*)calloc(total_neurons * max_incoming, sizeof(*cl_delays));
    uint16_t* cl_incoming =
        (uint16_t*)calloc(total_neurons, sizeof(*cl_incoming));
    uint16_t* cl_incoming_ids = (uint16_t*)calloc(total_neurons * max_incoming,
                                                  sizeof(*cl_incoming_ids));
    uint8_t* cl_is_input =
        (uint8_t*)calloc(total_neurons, sizeof(*cl_is_input));
    uint8_t* cl_is_output =
        (uint8_t*)calloc(total_neurons, sizeof(*cl_is_output));
    BwdParams bwd_params = {
        .v_decay            = leak,
        .v_rest             = (short)(min_potential / scale_factor),
        .tau_rho            = tau,
        .scale_rho          = rho,
        .num_neurons        = (ushort)total_neurons,
        .num_output_neurons = (ushort)output_neurons,
        .num_steps          = (short)timesteps,
        .max_incoming       = max_incoming,
        .scale_factor       = scale_factor,
    };

    // Current cpu-only allocations
    vector<vector<double>> weights(total_neurons);
    vector<vector<int>> delays(total_neurons);
    vector<double> thresholds(total_neurons);
    vector<vector<double>> m_weights(total_neurons);
    vector<vector<double>> v_weights(total_neurons);
    vector<vector<double>> delta_W(total_neurons);
    double b1_t = 1.0;
    double b2_t = 1.0;

    for (size_t i = 0; i < total_neurons; i++) {
        Node* nd = n->get_node(i);

        thresholds[i] =
            nd->get("Threshold") * ((discrete) ? scale_factor : 1.0);
        weights[i].reserve(nd->incoming.size());
        delays[i].reserve(nd->incoming.size());

        m_weights[i].resize(nd->incoming.size());
        v_weights[i].resize(nd->incoming.size());
        delta_W[i].resize(nd->incoming.size());

        {
            // OpenCL
            cl_thresh[i]    = nd->get("Threshold");
            cl_incoming[i]  = nd->incoming.size();
            cl_is_input[i]  = i < input_neurons;
            cl_is_output[i] = nd->is_output();
            ;
        }

        for (size_t j = 0; j < nd->incoming.size(); j++) {
            Edge* e = nd->incoming[j];

            weights[i].push_back(e->get("Weight") *
                                 ((discrete) ? scale_factor : 1.0));
            delays[i].push_back(e->get("Delay"));

            {
                // OpenCL
                cl_weights[i * max_incoming + j]      = e->get("Weight");
                cl_delays[i * max_incoming + j]       = e->get("Delay");
                cl_incoming_ids[i * max_incoming + j] = e->from->id;
            }
        }
    }

    cl_int err;
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetPlatformIDs failed: ");
        opencl_perror(err);
        exit(1);
    }

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetDeviceIDs failed for GPU: ");
        opencl_perror(err);
        exit(1);
    }
    if (device == NULL) {
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "clGetDeviceIDs failed for CPU: ");
            opencl_perror(err);
            exit(1);
        }
    }

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateContext failed: ");
        opencl_perror(err);
        exit(1);
    }
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateCommandQueue failed: ");
        opencl_perror(err);
        exit(1);
    }

    char* src              = load_kernel("kernels/risp.cl");
    const char* sources[1] = {src};

    cl_program prog = clCreateProgramWithSource(ctx, 1, sources, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateProgramWithSource failed: ");
        opencl_perror(err);
        exit(1);
    }
    err = clBuildProgram(prog, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clBuildProgram failed: ");
        opencl_perror(err);

        size_t log_size;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                              &log_size);
        char* log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log,
                              NULL);
        log[log_size] = '\0';
        fprintf(stderr, "%s\n", log);
        free(log);

        exit(1);
    }

    cl_kernel encode = clCreateKernel(prog, "RispEncodeInputs", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateKernel for RispEncodeInputs failed: ");
        opencl_perror(err);
        exit(1);
    }

    cl_kernel fwd = clCreateKernel(prog, "RispDynamicsFwdKernel", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateKernel for RispDynamicsFwdKernel failed: ");
        opencl_perror(err);
        exit(1);
    }

    cl_kernel spike_loss = clCreateKernel(prog, "RispSpikeLoss", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateKernel for RispSpikeLoss failed: ");
        opencl_perror(err);
        exit(1);
    }

    cl_kernel bwd = clCreateKernel(prog, "RispDynamicsBwdKernel", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateKernel for RispDynamicsBwdKernel failed: ");
        opencl_perror(err);
        exit(1);
    }

    cl_kernel w_updates = clCreateKernel(prog, "RispWeightUpdates", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateKernel for RispWeightUpdates failed: ");
        opencl_perror(err);
        exit(1);
    }

    free(src);

    cl_mem x_buf = clCreateBuffer(
        ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        input_neurons * timesteps * sizeof(cl_short), NULL, &err);
    cl_mem v_buf =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                       total_neurons * sizeof(cl_short), NULL, &err);
    cl_mem s_buf =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                       total_neurons * timesteps * sizeof(cl_char), NULL, &err);
    cl_mem v_pre_buf = clCreateBuffer(
        ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        total_neurons * timesteps * sizeof(cl_short), NULL, &err);
    cl_mem thresh_buf =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       total_neurons * sizeof(cl_short), cl_thresh, &err);
    cl_mem weights_buf = clCreateBuffer(
        ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        total_neurons * max_incoming * sizeof(cl_short), cl_weights, &err);
    cl_mem delays_buf = clCreateBuffer(
        ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        total_neurons * max_incoming * sizeof(cl_ushort), cl_delays, &err);
    cl_mem incoming_buf =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       total_neurons * sizeof(cl_ushort), cl_incoming, &err);
    cl_mem incoming_ids_buf =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       total_neurons * max_incoming * sizeof(cl_ushort),
                       cl_incoming_ids, &err);
    cl_mem is_input_buf =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       total_neurons * sizeof(cl_uchar), cl_is_input, &err);
    cl_mem is_output_buf =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       total_neurons * sizeof(cl_uchar), cl_is_output, &err);
    cl_mem train_data_buf =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       train.observations * train.cols * 2 * sizeof(cl_double),
                       train_encoded, &err);
    cl_mem dL_ds_buf =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                       output_neurons * sizeof(cl_double), NULL, &err);
    cl_mem spike_grad_history_buf = clCreateBuffer(
        ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        total_neurons * timesteps * sizeof(cl_double), NULL, &err);
    cl_mem voltage_grad_history_buf = clCreateBuffer(
        ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        total_neurons * timesteps * sizeof(cl_double), NULL, &err);
    cl_mem future_mem_grad_buf =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                       total_neurons * sizeof(cl_double), NULL, &err);
    cl_mem delta_W_buf = clCreateBuffer(
        ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        total_neurons * max_incoming * sizeof(cl_double), NULL, &err);
    cl_mem m_weights_buf = clCreateBuffer(
        ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        total_neurons * max_incoming * sizeof(cl_double), NULL, &err);
    cl_mem v_weights_buf = clCreateBuffer(
        ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        total_neurons * max_incoming * sizeof(cl_double), NULL, &err);
    cl_mem correct_buf =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                       1 * sizeof(cl_double), NULL, &err);
    cl_mem loss_buf =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                       1 * sizeof(cl_double), NULL, &err);
    cl_mem bwd_params_buf =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(BwdParams), &bwd_params, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer for bwd_params failed: ");
        opencl_perror(err);
        exit(1);
    }

    size_t* batch_order =
        (size_t*)calloc(train.observations, sizeof(*batch_order));

    {
        size_t encoder_work_size   = input_neurons;
        size_t fwd_work_size       = total_neurons;
        size_t loss_work_size      = output_neurons;
        size_t bwd_work_size       = total_neurons;
        size_t w_updates_work_size = total_neurons * max_incoming;

        int iinput_neurons          = input_neurons;
        int itimesteps              = timesteps;
        int16_t ispike_value_factor = spike_value_factor;
        short neuron_decay          = leak;
        short imin_potential        = min_potential / scale_factor;
        ushort inum_neurons         = total_neurons;
        ushort inum_steps           = timesteps;
        ushort imax_incoming        = max_incoming;
        ushort inum_output_neurons  = output_neurons;
        ushort ibatch_size          = batch_size;
        uint iepoch                 = 0;
        double beta1                = BETA1;
        double beta2                = BETA2;
        short imin_weight           = min_weight;
        short imax_weight           = max_weight;

        cl_event event_prev;
        cl_event event_curr;

        // Set kernel arguments that don't change
        {
            // Encode
            clSetKernelArg(encode, 0, sizeof(cl_mem), &x_buf);
            clSetKernelArg(encode, 1, sizeof(cl_mem), &train_data_buf);
            clSetKernelArg(encode, 2, sizeof(int), &train.cols);
            clSetKernelArg(encode, 3, sizeof(int), &iinput_neurons);
            clSetKernelArg(encode, 4, sizeof(int), &itimesteps);
            clSetKernelArg(encode, 6, sizeof(int16_t), &ispike_value_factor);

            // Forward
            clSetKernelArg(fwd, 0, sizeof(cl_mem), &x_buf);
            clSetKernelArg(fwd, 1, sizeof(cl_mem), &thresh_buf);
            clSetKernelArg(fwd, 2, sizeof(cl_mem), &weights_buf);
            clSetKernelArg(fwd, 3, sizeof(cl_mem), &delays_buf);
            clSetKernelArg(fwd, 4, sizeof(cl_mem), &incoming_buf);
            clSetKernelArg(fwd, 5, sizeof(cl_mem), &incoming_ids_buf);
            clSetKernelArg(fwd, 6, sizeof(cl_short), &neuron_decay);
            clSetKernelArg(fwd, 7, sizeof(cl_short), &imin_potential);
            clSetKernelArg(fwd, 8, sizeof(cl_mem), &is_input_buf);
            clSetKernelArg(fwd, 9, sizeof(cl_mem), &v_buf);
            clSetKernelArg(fwd, 10, sizeof(cl_mem), &s_buf);
            clSetKernelArg(fwd, 11, sizeof(cl_mem), &v_pre_buf);
            clSetKernelArg(fwd, 12, sizeof(cl_ushort), &inum_neurons);
            clSetKernelArg(fwd, 13, sizeof(cl_ushort), &inum_steps);
            clSetKernelArg(fwd, 15, sizeof(cl_ushort), &imax_incoming);

            // Loss
            clSetKernelArg(spike_loss, 0, sizeof(cl_mem), &dL_ds_buf);
            clSetKernelArg(spike_loss, 1, sizeof(cl_mem), &correct_buf);
            clSetKernelArg(spike_loss, 2, sizeof(cl_mem), &loss_buf);
            clSetKernelArg(spike_loss, 3, sizeof(cl_mem), &s_buf);
            clSetKernelArg(spike_loss, 4, sizeof(cl_ushort), &inum_neurons);
            clSetKernelArg(spike_loss, 5, sizeof(cl_ushort),
                           &inum_output_neurons);
            clSetKernelArg(spike_loss, 6, sizeof(cl_ushort), &inum_steps);

            // Backward
            clSetKernelArg(bwd, 0, sizeof(cl_mem), &dL_ds_buf);
            clSetKernelArg(bwd, 1, sizeof(cl_mem), &s_buf);
            clSetKernelArg(bwd, 2, sizeof(cl_mem), &v_pre_buf);
            clSetKernelArg(bwd, 3, sizeof(cl_mem), &thresh_buf);
            clSetKernelArg(bwd, 4, sizeof(cl_mem), &is_output_buf);
            clSetKernelArg(bwd, 5, sizeof(cl_mem), &spike_grad_history_buf);
            clSetKernelArg(bwd, 6, sizeof(cl_mem), &voltage_grad_history_buf);
            clSetKernelArg(bwd, 7, sizeof(cl_mem), &future_mem_grad_buf);
            clSetKernelArg(bwd, 8, sizeof(cl_mem), &delta_W_buf);
            clSetKernelArg(bwd, 9, sizeof(cl_mem), &weights_buf);
            clSetKernelArg(bwd, 10, sizeof(cl_mem), &delays_buf);
            clSetKernelArg(bwd, 11, sizeof(cl_mem), &incoming_buf);
            clSetKernelArg(bwd, 12, sizeof(cl_mem), &incoming_ids_buf);
            clSetKernelArg(bwd, 13, sizeof(cl_mem), &bwd_params_buf);

            // Weight updates
            clSetKernelArg(w_updates, 0, sizeof(cl_mem), &m_weights_buf);
            clSetKernelArg(w_updates, 1, sizeof(cl_mem), &v_weights_buf);
            clSetKernelArg(w_updates, 2, sizeof(cl_mem), &delta_W_buf);
            clSetKernelArg(w_updates, 3, sizeof(cl_mem), &weights_buf);
            clSetKernelArg(w_updates, 4, sizeof(cl_mem), &incoming_buf);
            clSetKernelArg(w_updates, 5, sizeof(cl_ushort), &inum_neurons);
            clSetKernelArg(w_updates, 6, sizeof(cl_ushort), &imax_incoming);
            clSetKernelArg(w_updates, 7, sizeof(cl_double), &learning_rate);
            clSetKernelArg(w_updates, 8, sizeof(cl_double), &decay_rate);
            clSetKernelArg(w_updates, 10, sizeof(cl_ushort), &ibatch_size);
            clSetKernelArg(w_updates, 13, sizeof(cl_double), &beta1);
            clSetKernelArg(w_updates, 14, sizeof(cl_double), &beta2);
            clSetKernelArg(w_updates, 17, sizeof(cl_ushort), &inum_steps);
            clSetKernelArg(w_updates, 18, sizeof(cl_uint), &train.observations);
            clSetKernelArg(w_updates, 19, sizeof(cl_double), &scale_factor);
            clSetKernelArg(w_updates, 20, sizeof(cl_short), &imin_weight);
            clSetKernelArg(w_updates, 21, sizeof(cl_short), &imax_weight);
            clSetKernelArg(w_updates, 22, sizeof(cl_int), &scale);
        }

        for (size_t epoch = 0; epoch < epochs; epoch++) {
            double epoch_loss    = 0.0;
            double epoch_correct = 0.0;

            for (int i = 0; i < train.observations; i++) {
                batch_order[i] = i;
            }

            // Shuffle batch order
            for (int i = 0; i < train.observations; i++) {
                int j          = rand() % train.observations;
                size_t tmp     = batch_order[i];
                batch_order[i] = batch_order[j];
                batch_order[j] = tmp;
            }

            for (int batch_start = 0; batch_start < train.observations;
                 batch_start += batch_size) {
                size_t current_batch_size =
                    min((size_t)batch_size,
                        train.observations - (size_t)batch_start);

                event_prev = NULL;
                event_curr = NULL;

                for (size_t i = 0; i < current_batch_size; i++) {
                    int observation_idx = batch_order[batch_start + i];

                    if (event_curr) {
                        clWaitForEvents(1, &event_curr);
                    }

                    // Zero buffers
                    opencl_zerobuf(queue, x_buf, input_neurons * timesteps,
                                   sizeof(cl_short), &event_curr);
                    opencl_zerobuf(queue, v_buf, total_neurons,
                                   sizeof(cl_short), &event_curr);
                    opencl_zerobuf(queue, s_buf, total_neurons * timesteps,
                                   sizeof(cl_char), &event_curr);
                    opencl_zerobuf(queue, v_pre_buf, total_neurons * timesteps,
                                   sizeof(cl_short), &event_curr);
                    opencl_zerobuf(queue, dL_ds_buf, output_neurons,
                                   sizeof(cl_double), &event_curr);
                    opencl_zerobuf(queue, correct_buf, 1, sizeof(cl_double),
                                   &event_curr);
                    opencl_zerobuf(queue, loss_buf, 1, sizeof(cl_double),
                                   &event_curr);
                    opencl_zerobuf(queue, spike_grad_history_buf,
                                   total_neurons * timesteps, sizeof(cl_float),
                                   &event_curr);
                    opencl_zerobuf(queue, voltage_grad_history_buf,
                                   total_neurons * timesteps, sizeof(cl_float),
                                   &event_curr);
                    opencl_zerobuf(queue, future_mem_grad_buf, total_neurons,
                                   sizeof(cl_float), &event_curr);

                    event_prev = event_curr;

                    clSetKernelArg(encode, 5, sizeof(int), &observation_idx);
                    err = clEnqueueNDRangeKernel(queue, encode, 1, NULL,
                                                 &encoder_work_size, NULL, 1,
                                                 &event_prev, &event_curr);
                    if (err != CL_SUCCESS) {
                        fprintf(stderr,
                                "clEnqueueNDRangeKernel for RispEncodeInputs "
                                "failed: ");
                        opencl_perror(err);
                        exit(1);
                    }

                    event_prev = event_curr;

                    // Forward pass
                    for (short t = 0; t < (short)timesteps; t++) {
                        clSetKernelArg(fwd, 14, sizeof(cl_ushort), &t);

                        err = clEnqueueNDRangeKernel(queue, fwd, 1, NULL,
                                                     &fwd_work_size, NULL, 1,
                                                     &event_prev, &event_curr);
                        if (err != CL_SUCCESS) {
                            fprintf(stderr,
                                    "clEnqueueNDRangeKernel for "
                                    "RispDynamicsFwdKernel "
                                    "failed @t=%hd: ",
                                    t);
                            opencl_perror(err);
                            exit(1);
                        }

                        event_prev = event_curr;
                    }

                    // Loss kernel
                    ushort target_idx = train.labels[observation_idx];
                    clSetKernelArg(spike_loss, 7, sizeof(cl_ushort),
                                   &target_idx);
                    err = clEnqueueNDRangeKernel(queue, spike_loss, 1, NULL,
                                                 &loss_work_size, NULL, 1,
                                                 &event_prev, &event_curr);
                    if (err != CL_SUCCESS) {
                        fprintf(stderr, "clEnqueueNDRangeKernel for "
                                        "RispSpikeLoss failed: ");
                        opencl_perror(err);
                        exit(1);
                    }

                    event_prev = event_curr;

                    // Read loss and correct count
                    double loss;
                    double correct;
                    if (epoch % 10 == 0) {
                        err = clEnqueueReadBuffer(queue, loss_buf, CL_TRUE, 0,
                                                  sizeof(double), &loss, 1,
                                                  &event_prev, &event_curr);
                        event_prev = event_curr;
                        err = clEnqueueReadBuffer(queue, correct_buf, CL_TRUE,
                                                  0, sizeof(double), &correct,
                                                  1, &event_prev, &event_curr);
                        event_prev = event_curr;

                        epoch_loss += loss;
                        epoch_correct += correct;
                    }

                    // Backward pass
                    for (short t = (short)timesteps - 1; t >= 0; t--) {
                        clSetKernelArg(bwd, 14, sizeof(cl_short), &t);

                        err = clEnqueueNDRangeKernel(queue, bwd, 1, NULL,
                                                     &bwd_work_size, NULL, 1,
                                                     &event_prev, &event_curr);
                        if (err != CL_SUCCESS) {
                            fprintf(stderr,
                                    "clEnqueueNDRangeKernel for "
                                    "RispDynamicsBwdKernel "
                                    "failed @t=%hd: ",
                                    t);
                            opencl_perror(err);
                            exit(1);
                        }

                        event_prev = event_curr;
                    }
                }

                // Weight updates
                ushort icurrent_batch_size = current_batch_size;
                uint ibatch_start          = batch_start;
                iepoch                     = epoch;
                b1_t *= BETA1;
                b2_t *= BETA2;
                clSetKernelArg(w_updates, 9, sizeof(cl_ushort),
                               &icurrent_batch_size);
                clSetKernelArg(w_updates, 11, sizeof(cl_uint), &ibatch_start);
                clSetKernelArg(w_updates, 12, sizeof(cl_uint), &iepoch);
                clSetKernelArg(w_updates, 15, sizeof(cl_double), &b1_t);
                clSetKernelArg(w_updates, 16, sizeof(cl_double), &b2_t);
                err = clEnqueueNDRangeKernel(queue, w_updates, 1, NULL,
                                             &w_updates_work_size, NULL, 1,
                                             &event_prev, &event_curr);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "clEnqueueNDRangeKernel for "
                                    "RispWeightUpdates failed: ");
                    opencl_perror(err);
                    exit(1);
                }

                clWaitForEvents(1, &event_curr);
            }

            clFinish(queue);

            printf("Epoch %zu: Train Loss: %8g, Train Acc: %8g\n", epoch,
                   epoch_loss / (double)train.observations,
                   epoch_correct / (double)train.observations);
        }
    }

    exit(1);

    ThreadArgs* tas          = (ThreadArgs*)calloc(threads, sizeof(*tas));
    pthread_t* tids          = (pthread_t*)calloc(threads, sizeof(*tids));
    int max_idx              = -1;
    pthread_mutex_t mut      = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t have_work = PTHREAD_COND_INITIALIZER;
    pthread_cond_t done_work = PTHREAD_COND_INITIALIZER;
    bool train_p             = true;
    bool die                 = false;
    int work_idx             = 0;
    int done_count           = 0;

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

        printf("Epoch: %4zu/%zu, Loss: %10g (Best: %10g), Acc: %10g (Best: "
               "%10g), "
               "TestLoss: %10g (Best: %10g), TestAcc: %10g (Best: %10g)\n",
               epoch + 1, epochs, epoch_loss / (double)train.observations,
               best_train_loss, correct / (double)train.observations,
               best_train_acc, test_loss, best_test_loss, test_correct,
               best_test_acc);

        if (network_json_out) {
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
