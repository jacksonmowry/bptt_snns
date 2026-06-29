#include "csv.h"
#include "framework.hpp"
#include <CL/cl.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <getopt.h>
#include <stddef.h>
#include <unordered_set>

using namespace std;
using namespace neuro;
using nlohmann::json;

// Adam/Learning Parameters
#define BETA1 (0.9)
#define BETA2 (0.999)
#define ADAM_EPS (1.0e-8)
#define NUM_LAYERS (3)

struct TrainingBundle {
    vector<vector<double>> weights;
    vector<vector<double>> delta_W;
    vector<vector<int>> delays;
    vector<double> thresholds;
    vector<vector<double>> m_weights;
    vector<vector<double>> v_weights;
    vector<vector<double>> spikes;
    vector<vector<double>> v_pre;
    vector<double> spike_logits;
    vector<double> target;
    vector<double> dL_ds;
    vector<double> softmax_out;

    Eigen::VectorXd future_mem_grad_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> sgh;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> vgh;
    Eigen::VectorXd dL_dV_;
    Eigen::VectorXd v_pre_t_;
    Eigen::VectorXd dV_post_dV_pre_;
    Eigen::VectorXd dV_post_ds_t_;
    Eigen::VectorXd ds_t_dV_pre_;
    Eigen::VectorXd dV_leak_dV_t1_;
    Eigen::VectorXd grad_;

    double b1_t = 1.0;
    double b2_t = 1.0;

    double tau;
    double rho;

    double learning_rate;
    double decay_rate;

    TrainingBundle(size_t total_neurons, size_t timesteps,
                   size_t output_neurons, double tau, double rho,
                   double learning_rate, double decay_rate)
        : weights(total_neurons), delta_W(total_neurons), delays(total_neurons),
          thresholds(total_neurons), m_weights(total_neurons),
          v_weights(total_neurons),
          spikes(timesteps, vector<double>(total_neurons)),
          v_pre(timesteps, vector<double>(total_neurons)),
          spike_logits(output_neurons), target(output_neurons),
          dL_ds(output_neurons), softmax_out(output_neurons),
          future_mem_grad_(total_neurons), sgh(total_neurons, timesteps),
          vgh(total_neurons, timesteps), dL_dV_(total_neurons),
          v_pre_t_(total_neurons), dV_post_dV_pre_(total_neurons),
          dV_post_ds_t_(total_neurons), ds_t_dV_pre_(total_neurons),
          dV_leak_dV_t1_(total_neurons), grad_(total_neurons), tau(tau),
          rho(rho), learning_rate(learning_rate), decay_rate(decay_rate) {}
};

struct EvaluationResults {
    double correct;
    double loss;
};

struct NetworkConfiguration {
    Network* n;

    int input_neurons;
    int hidden_neurons;
    int output_neurons;
    int layer_offsets[3];
    int total_neurons;

    int timesteps;
    bool timeseries;

    double min_potential;
    bool leak;
};

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

double normal(double mean, double stddev) {
    double u1, u2;
    do {
        u1 = drand48();
    } while (u1 == 0.0); // Avoid log(0)
    u2       = drand48();
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * acos(-1.0) * u2);
    return mean + stddev * z;
}

void softmax(const double* logits, double* out, size_t n) {
    double max = logits[0];
    for (size_t i = 1; i < n; i++) {
        if (logits[i] > max) {
            max = logits[i];
        }
    }

    double exp_sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        out[i] = exp(logits[i] - max);
        exp_sum += out[i];
    }

    for (size_t i = 0; i < n; i++) {
        out[i] /= exp_sum;
    }
}

double cross_entropy(const double* logits, const double* targets, double* grads,
                     size_t n) {
    softmax(logits, grads, n);

    double loss = 0.0;
    for (size_t i = 0; i < n; i++) {
        loss -= targets[i] * log(grads[i] + ADAM_EPS);
        grads[i] -= targets[i];
    }

    return loss;
}

double alpha(bool leak) { return (double)!leak; }
float decay_f(bool leak) { return (double)leak; }

double spike_surrogate(double v_pre_t, double v_thresh, double scale_rho,
                       double tau_rho_scaled) {
    return (scale_rho / (2.0 * tau_rho_scaled)) *
           expf(-fabs(v_pre_t - v_thresh) / tau_rho_scaled);
}

size_t label_count(const Dataset* d) {
    unordered_set<double> us;
    for (int i = 0; i < d->observations; i++) {
        us.insert(d->labels[i]);
    }

    return us.size();
}

void preprocess_data(Dataset* d) {
    assert(d->rows_per_observation == -1);
    for (int i = 0; i < d->observations; i++) {
        for (int j = 0; j < d->cols; j++) {
            double x     = (d->data[i * d->cols + j] - d->min_vals[j]) /
                           (d->max_vals[j] - d->min_vals[j]);
            double inv_x = 1.0 - x;

            d->processed_data[i * (d->cols * 2) + j * 2] =
                (x > 0.0) ? 1.0 / x : 0.0;
            d->processed_data[i * (d->cols * 2) + j * 2 + 1] =
                (inv_x > 0.0) ? 1.0 / inv_x : 0.0;
        }
    }
}

void encode_spikes(Processor* p, const Dataset* d, size_t index,
                   size_t timesteps, bool timeseries, size_t input_neurons,
                   int* encoded_spikes) {
    if (timeseries) {
        size_t encoding_window = timesteps / d->cols;
        assert(encoding_window > 0);

        for (size_t input = 0; input < input_neurons / 2; input++) {
            for (int column_t = 0; column_t < d->cols; column_t++) {
                float encoding_start = column_t * encoding_window;
                float encoding_end   = encoding_start + encoding_window;

                float x = (d->data[(index * d->rows_per_observation * d->cols) +
                                   (input * d->cols) + column_t] -
                           d->min_vals[input]) /
                          (d->max_vals[input] - d->min_vals[input]);
                float inv_x = 1.0 - x;

                if (x > 0.0) {
                    for (float j = encoding_start; j < encoding_end;
                         j += 1.0 / x) {
                        p->apply_spike({(int)input * 2, (float)(int)j, 1.0});
                    }
                }
                if (inv_x > 0.0) {
                    for (float j = encoding_start; j < encoding_end;
                         j += 1.0 / inv_x) {
                        p->apply_spike(
                            {(int)input * 2 + 1, (float)(int)j, 1.0});
                    }
                }
            }
        }
    } else {
        for (size_t input = 0; input < input_neurons / 2; input++) {
            double x = (d->data[index * d->cols + input] - d->min_vals[input]) /
                       (d->max_vals[input] - d->min_vals[input]);
            double inv_x = 1.0 - x;
            if (x > 0.0f) {
                x = 1.0f / x;
                for (float j = 0; j < (float)timesteps; j += x) {
                    p->apply_spike({(int)input * 2, (double)(size_t)j, 1.0});
                    encoded_spikes[((input * 2) * timesteps) + (int)j] = 1;
                }
            }
            if (inv_x > 0.0f) {
                inv_x = 1.0f / inv_x;
                for (float j = 0; j < (float)timesteps; j += inv_x) {
                    p->apply_spike(
                        {(int)input * 2 + 1, (double)(size_t)j, 1.0});
                    encoded_spikes[((input * 2 + 1) * timesteps) + (int)j] = 1;
                }
            }
        }
    }
}

EvaluationResults forward(TrainingBundle* tb, Processor* p, const Dataset* d,
                          size_t index, const NetworkConfiguration* nc,
                          int* encoded_spikes) {
    EvaluationResults er = {0.0, 0.0};

    p->clear_activity();

    for (int t = 0; t < nc->timesteps; t++) {
        fill(tb->spikes[t].begin(), tb->spikes[t].end(), 0.0);
        fill(tb->v_pre[t].begin(), tb->v_pre[t].end(), 0.0);
    }
    fill(tb->spike_logits.begin(), tb->spike_logits.end(), 0.0);

    encode_spikes(p, d, index, nc->timesteps, nc->timeseries, nc->input_neurons,
                  encoded_spikes);

    for (int t = 0; t < nc->timesteps; t++) {
        p->run(1);

        const vector<int>& neuron_counts         = p->neuron_counts();
        const vector<double>& neuron_pre_charges = p->neuron_pre_charges();
        for (int neuron = 0; neuron < nc->total_neurons; neuron++) {
            tb->spikes[t][neuron] = neuron_counts[neuron];
            tb->v_pre[t][neuron]  = neuron_pre_charges[neuron];

            if (neuron >= nc->layer_offsets[2]) {
                tb->spike_logits[neuron - nc->layer_offsets[2]] +=
                    neuron_counts[neuron];
            }
        }
    }

    size_t max_idx = 0;
    double max_val = 0;
    for (int neuron = 0; neuron < nc->output_neurons; neuron++) {
        tb->spike_logits[neuron] /= (double)nc->timesteps;

        if (tb->spike_logits[neuron] > max_val) {
            max_idx = neuron;
            max_val = tb->spike_logits[neuron];
        }
    }

    if (max_idx == (size_t)d->labels[index]) {
        er.correct++;
    }

    for (int i = 0; i < nc->output_neurons; i++) {
        if (i == (int)d->labels[index]) {
            tb->target[i] = 1.0;
        } else {
            tb->target[i] = 0.0;
        }
    }

    double loss_spike =
        cross_entropy(tb->spike_logits.data(), tb->target.data(),
                      tb->dL_ds.data(), nc->output_neurons);
    er.loss = loss_spike;

    return er;
}

void backward(TrainingBundle* tb, const NetworkConfiguration* nc) {
    tb->future_mem_grad_.setZero();
    tb->sgh.setZero();
    tb->vgh.setZero();
    tb->dL_dV_.setZero();
    tb->v_pre_t_.setZero();
    tb->dV_post_dV_pre_.setZero();
    tb->dV_post_ds_t_.setZero();
    tb->ds_t_dV_pre_.setZero();
    tb->dV_leak_dV_t1_.setZero();
    tb->grad_.setZero();

    for (int t = nc->timesteps - 1; t >= 0; t--) {
        tb->sgh.col(t).segment(nc->layer_offsets[2], nc->output_neurons) +=
            Eigen::Map<const Eigen::VectorXd>(&tb->dL_ds[0],
                                              nc->output_neurons) /
            nc->timesteps;

        tb->dL_dV_   = tb->vgh.col(t) + tb->future_mem_grad_;
        tb->v_pre_t_ = Eigen::Map<const Eigen::VectorXd>(&tb->v_pre[t][0],
                                                         nc->total_neurons);

        tb->dV_post_dV_pre_ = (Eigen::Map<const Eigen::VectorXd>(
                                   &tb->spikes[t][0], nc->total_neurons)
                                   .array() <= 0)
                                  .cast<double>();

        tb->dV_post_ds_t_ = -tb->v_pre_t_;
        if (nc->min_potential > 0) {
            (tb->dV_post_ds_t_.array() + nc->min_potential).matrix();
        }

        tb->ds_t_dV_pre_ =
            (tb->rho / (2.0 * tb->tau)) *
            (-(tb->v_pre_t_ - Eigen::Map<const Eigen::VectorXd>(
                                  &tb->thresholds[0], nc->total_neurons))
                  .array()
                  .abs()
                  .matrix() /
             tb->tau)
                .array()
                .exp()
                .matrix();

        tb->dV_leak_dV_t1_ =
            (tb->v_pre_t_.array() >= nc->min_potential).cast<double>() *
            (1.0 - nc->leak);

        tb->grad_ = (tb->dL_dV_.array() * tb->dV_post_dV_pre_.array()) +
                    (tb->dL_dV_.array() * tb->dV_post_ds_t_.array() *
                     tb->ds_t_dV_pre_.array()) +
                    (tb->sgh.col(t).array() * tb->ds_t_dV_pre_.array());

        tb->future_mem_grad_ =
            (tb->dL_dV_.array() * tb->dV_post_dV_pre_.array() *
             tb->dV_leak_dV_t1_.array()) +
            (tb->dL_dV_.array() * tb->dV_post_ds_t_.array() *
             tb->ds_t_dV_pre_.array() * tb->dV_leak_dV_t1_.array()) +
            (tb->sgh.col(t).array() * tb->ds_t_dV_pre_.array() *
             tb->dV_leak_dV_t1_.array());

        for (int dest = nc->total_neurons - 1; dest >= 0; dest--) {
            for (size_t source_idx = 0;
                 source_idx < nc->n->get_node(dest)->incoming.size();
                 source_idx++) {
                size_t source =
                    nc->n->get_node(dest)->incoming[source_idx]->from->id;

                int delay       = tb->delays[dest][source_idx];
                int source_time = t - delay;
                if (source_time < 0) {
                    continue;
                }

                double source_spike = tb->spikes[source_time][source];
                tb->delta_W[dest][source_idx] += source_spike * tb->grad_(dest);
                tb->sgh(source, source_time) +=
                    tb->grad_(dest) * tb->weights[dest][source_idx];
            }
        }
    }
}

void weight_updates(TrainingBundle* tb, const NetworkConfiguration* nc,
                    const Dataset* d, size_t current_batch_size,
                    size_t batch_size, size_t batch_start, size_t epoch) {
    double inv_batch = 1.0 / ((double)current_batch_size * nc->timesteps);

    tb->b1_t *= BETA1;
    tb->b2_t *= BETA2;

    for (int i = 0; i < nc->total_neurons; i++) {
        for (size_t j = 0; j < nc->n->get_node(i)->incoming.size(); j++) {
            Edge* e = nc->n->get_node(i)->incoming[j];

            tb->delta_W[i][j] *= inv_batch;

            tb->m_weights[i][j] =
                BETA1 * tb->m_weights[i][j] + (1.0 - BETA1) * tb->delta_W[i][j];
            tb->v_weights[i][j] =
                BETA2 * tb->v_weights[i][j] +
                (1.0 - BETA2) * (tb->delta_W[i][j] * tb->delta_W[i][j]);
            tb->delta_W[i][j] = 0.0;

            double mW_hat = tb->m_weights[i][j] / (1.0 - tb->b1_t);
            double vW_hat = tb->v_weights[i][j] / (1.0 - tb->b2_t);

            double lr = tb->learning_rate;
            if (epoch == 0) {
                lr = ((batch_start + batch_size) / (double)d->observations) *
                     tb->learning_rate;
            }

            tb->weights[i][j] -= lr * mW_hat / (sqrt(vW_hat + ADAM_EPS));
            tb->weights[i][j] -= lr * tb->decay_rate * tb->weights[i][j];
            e->set("Weight", tb->weights[i][j]);
        }
    }
}

char* load_kernel(const char* path) {
    FILE* f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "Unable to open kernel for reading: %s\n", path);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* src = (char*)malloc(len + 1);
    fread(src, 1, len, f);
    src[len] = '\0';
    fclose(f);
    return src;
}

void opencl_perror(int err) {
    switch (err) {
    case 0:
        puts("CL_SUCCESS");
        break;
    case -1:
        puts("CL_DEVICE_NOT_FOUND                        ");
        break;
    case -2:
        puts("CL_DEVICE_NOT_AVAILABLE                    ");
        break;
    case -3:
        puts("CL_COMPILER_NOT_AVAILABLE                  ");
        break;
    case -4:
        puts("CL_MEM_OBJECT_ALLOCATION_FAILURE           ");
        break;
    case -5:
        puts("CL_OUT_OF_RESOURCES                        ");
        break;
    case -6:
        puts("CL_OUT_OF_HOST_MEMORY                      ");
        break;
    case -7:
        puts("CL_PROFILING_INFO_NOT_AVAILABLE            ");
        break;
    case -8:
        puts("CL_MEM_COPY_OVERLAP                        ");
        break;
    case -9:
        puts("CL_IMAGE_FORMAT_MISMATCH                   ");
        break;
    case -10:
        puts("CL_IMAGE_FORMAT_NOT_SUPPORTED              ");
        break;
    case -11:
        puts("CL_BUILD_PROGRAM_FAILURE                   ");
        break;
    case -12:
        puts("CL_MAP_FAILURE                             ");
        break;
    case -13:
        puts("CL_MISALIGNED_SUB_BUFFER_OFFSET            ");
        break;
    case -14:
        puts("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST");
        break;
    case -15:
        puts("CL_COMPILE_PROGRAM_FAILURE");
        break;
    case -16:
        puts("CL_LINKER_NOT_AVAILABLE                    ");
        break;
    case -17:
        puts("CL_LINK_PROGRAM_FAILURE                    ");
        break;
    case -18:
        puts("CL_DEVICE_PARTITION_FAILED                 ");
        break;
    case -19:
        puts("CL_KERNEL_ARG_INFO_NOT_AVAILABLE           ");
        break;
    case -30:
        puts("CL_INVALID_VALUE                           ");
        break;
    case -31:
        puts("CL_INVALID_DEVICE_TYPE                     ");
        break;
    case -32:
        puts("CL_INVALID_PLATFORM                        ");
        break;
    case -33:
        puts("CL_INVALID_DEVICE                          ");
        break;
    case -34:
        puts("CL_INVALID_CONTEXT                         ");
        break;
    case -35:
        puts("CL_INVALID_QUEUE_PROPERTIES                ");
        break;
    case -36:
        puts("CL_INVALID_COMMAND_QUEUE                   ");
        break;
    case -37:
        puts("CL_INVALID_HOST_PTR                        ");
        break;
    case -38:
        puts("CL_INVALID_MEM_OBJECT                      ");
        break;
    case -39:
        puts("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR         ");
        break;
    case -40:
        puts("CL_INVALID_IMAGE_SIZE                      ");
        break;
    case -41:
        puts("CL_INVALID_SAMPLER                         ");
        break;
    case -42:
        puts("CL_INVALID_BINARY                          ");
        break;
    case -43:
        puts("CL_INVALID_BUILD_OPTIONS                   ");
        break;
    case -44:
        puts("CL_INVALID_PROGRAM                         ");
        break;
    case -45:
        puts("CL_INVALID_PROGRAM_EXECUTABLE              ");
        break;
    case -46:
        puts("CL_INVALID_KERNEL_NAME                     ");
        break;
    case -47:
        puts("CL_INVALID_KERNEL_DEFINITION               ");
        break;
    case -48:
        puts("CL_INVALID_KERNEL                          ");
        break;
    case -49:
        puts("CL_INVALID_ARG_INDEX                       ");
        break;
    case -50:
        puts("CL_INVALID_ARG_VALUE                       ");
        break;
    case -51:
        puts("CL_INVALID_ARG_SIZE                        ");
        break;
    case -52:
        puts("CL_INVALID_KERNEL_ARGS                     ");
        break;
    case -53:
        puts("CL_INVALID_WORK_DIMENSION                  ");
        break;
    case -54:
        puts("CL_INVALID_WORK_GROUP_SIZE                 ");
        break;
    case -55:
        puts("CL_INVALID_WORK_ITEM_SIZE                  ");
        break;
    case -56:
        puts("CL_INVALID_GLOBAL_OFFSET                   ");
        break;
    case -57:
        puts("CL_INVALID_EVENT_WAIT_LIST                 ");
        break;
    case -58:
        puts("CL_INVALID_EVENT                           ");
        break;
    case -59:
        puts("CL_INVALID_OPERATION                       ");
        break;
    case -60:
        puts("CL_INVALID_GL_OBJECT                       ");
        break;
    case -61:
        puts("CL_INVALID_BUFFER_SIZE                     ");
        break;
    case -62:
        puts("CL_INVALID_MIP_LEVEL                       ");
        break;
    case -63:
        puts("CL_INVALID_GLOBAL_WORK_SIZE                ");
        break;
    case -64:
        puts("CL_INVALID_PROPERTY                        ");
        break;
    case -65:
        puts("CL_INVALID_IMAGE_DESCRIPTOR                ");
        break;
    case -66:
        puts("CL_INVALID_COMPILER_OPTIONS                ");
        break;
    case -67:
        puts("CL_INVALID_LINKER_OPTIONS                  ");
        break;
    case -68:
        puts("CL_INVALID_DEVICE_PARTITION_COUNT          ");
        break;
    }
}

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [OPTIONS]...\n", prog);
    fprintf(stderr, "  -n, --network_json     FILE    Network JSON path\n");
    fprintf(stderr, "  -d, --data_file        FILE    Data file path\n");
    fprintf(stderr, "  -l, --label_file       FILE    Label file path\n");
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
    fprintf(stderr, "  -h, --help                     Show this help\n");
}

int main(int argc, char* argv[]) {
    char* network_json_file = NULL;
    char* data_file         = NULL;
    char* label_file        = NULL;
    bool timeseries         = false;
    double connectivity     = 0.2;
    double learning_rate    = 0.008;
    double decay_rate       = 0.0001;
    double tau              = 0.95;
    double rho              = 1.4;
    int timesteps           = 32;
    int hidden_neurons      = 16;
    unsigned long seed      = (unsigned long)time(NULL);
    size_t epochs           = 10;
    size_t batch_size       = 1;
    double training_percent = 0.8;

    static struct option long_options[] = {
        {"network_json", required_argument, 0, 'n'},
        {"data_file", required_argument, 0, 'd'},
        {"label_file", required_argument, 0, 'l'},
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
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0},
    };

    int c;
    char* endptr;

    while ((c = getopt_long(argc, argv, "n:d:l:b:c:r:e:u:o:t:H:s:hp:B:P:",
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
        case 'h':
            print_usage(argv[0]);
            return 0;
        case '?':
            print_usage(argv[0]);
            return 1;
        }
    }

    if (network_json_file == NULL || data_file == NULL || label_file == NULL) {
        fprintf(stderr, "Error: --network_json, --data_file, and --label_file "
                        "are required.\n");
        print_usage(argv[0]);
        return 1;
    }

    srand(seed);
    srand48(seed);

    Dataset train;
    Dataset test;

    if (timeseries) {
        load_dataset_2d(data_file, label_file, training_percent, &train, &test);
    } else {
        load_dataset(data_file, label_file, training_percent, &train, &test);
    }
    train.processed_data = (float*)malloc(train.observations * train.cols * 2 *
                                          sizeof(*train.processed_data));
    test.processed_data  = (float*)malloc(test.observations * test.cols * 2 *
                                          sizeof(*test.processed_data));
    printf("Train rows: %d\n", train.observations);
    preprocess_data(&train);
    printf("Train rows: %d\n", train.observations);
    preprocess_data(&test);

    size_t train_labels = label_count(&train);
    size_t test_labels  = label_count(&test);
    assert(test.observations == 0 || train_labels == test_labels);

    int input_neurons =
        (timeseries) ? train.rows_per_observation * 2 : train.cols * 2;
    int output_neurons = train_labels;
    int total_neurons  = input_neurons + hidden_neurons + output_neurons;

    json emptynet;
    ifstream fin(network_json_file);
    fin >> emptynet;
    fin.close();

    Network* n = new Network();
    n->from_json(emptynet);

    string leak_prop         = n->get_data("proc_params")["leak_mode"];
    bool leak                = leak_prop == "all";
    float neuron_decay       = decay_f(leak);
    float spike_value_factor = n->get_data("proc_params")["spike_value_factor"];
    float min_potential      = n->get_data("proc_params")["min_potential"];

    if (!n) {
        fprintf(stderr, "%s:%s:%d: Unable to create network.\n", __FILE__,
                __FUNCTION__, __LINE__);
        exit(1);
    }

    const int layer_sizes[3] = {input_neurons, hidden_neurons, output_neurons};

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
    };

    size_t neuron_count  = 0;
    size_t synapse_count = 0;

    for (size_t i = 0; i < NUM_LAYERS; i++) {
        for (int j = 0; j < layer_sizes[i]; j++) {
            n->add_node(neuron_count)->set("Threshold", 1.0);

            if (i == 0) {
                // Input
                n->add_input(neuron_count);
            } else if (i == NUM_LAYERS - 1) {
                // Output
                n->add_output(neuron_count);
            }

            neuron_count++;
        }
    }

    int max_incoming = 0;
    for (int i = 0; i < total_neurons; i++) {
        int incoming = 0;
        for (int j = 0; j < total_neurons; j++) {
            if (drand48() < (1.0 - connectivity)) {
                continue;
            }

            Edge* e = n->add_edge(i, j);
            synapse_count++;
            incoming++;

            e->set(n->get_edge_property("Weight")->index, normal(0.0, 0.1));
            e->set(n->get_edge_property("Delay")->index, rand() % 7 + 1);
        }

        max_incoming = max(max_incoming, incoming);
    }

    printf("Neurons: %zu, Synapses: %zu\n", neuron_count, synapse_count);

    float* thresh = (float*)calloc(total_neurons, sizeof(*thresh));
    float* weights =
        (float*)calloc(total_neurons * max_incoming, sizeof(*weights));
    int* delays   = (int*)calloc(total_neurons * max_incoming, sizeof(*delays));
    int* incoming = (int*)calloc(total_neurons, sizeof(*incoming));
    int* incoming_ids =
        (int*)calloc(total_neurons * max_incoming, sizeof(*incoming_ids));
    int* is_input = (int*)calloc(total_neurons, sizeof(*is_input));

    TrainingBundle tb(total_neurons, timesteps, output_neurons, tau, rho,
                      learning_rate, decay_rate);

    for (int i = 0; i < total_neurons; i++) {
        tb.thresholds[i] = n->get_node(i)->get("Threshold");
        thresh[i]        = n->get_node(i)->get("Threshold");

        incoming[i] = n->get_node(i)->incoming.size();
        is_input[i] = i < input_neurons;

        tb.weights[i].reserve(n->get_node(i)->incoming.size());
        tb.delays[i].reserve(n->get_node(i)->incoming.size());
        tb.delta_W[i].resize(n->get_node(i)->incoming.size());
        tb.m_weights[i].resize(n->get_node(i)->incoming.size());
        tb.v_weights[i].resize(n->get_node(i)->incoming.size());

        for (size_t j = 0; j < n->get_node(i)->incoming.size(); j++) {
            Edge* e = n->get_node(i)->incoming[j];

            tb.weights[i].push_back(e->get("Weight"));
            tb.delays[i].push_back(e->get("Delay"));

            weights[i * max_incoming + j]      = e->get("Weight");
            delays[i * max_incoming + j]       = e->get("Delay");
            incoming_ids[i * max_incoming + j] = e->from->id;

            if (tb.weights[i].back() != weights[i * max_incoming + j]) {
                printf("Weights don't match %g & %g\n", tb.weights[i].back(),
                       weights[i * max_incoming + j]);
            }
            assert(tb.weights[i].back() == weights[i * max_incoming + j]);
        }
    }

    puts("Diff in tb.weights & weights");
    for (int i = 0; i < total_neurons; i++) {
        for (int j = 0; j < incoming[i]; j++) {
            printf("%g ", tb.weights[i][j] - weights[i * max_incoming + j]);
        }
        puts("");
    }
    puts("");
    exit(1);

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

    cl_kernel fwd = clCreateKernel(prog, "RispDynamicsFwdKernel", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateKernel for RispDynamicsFwdKernel failed: ");
        opencl_perror(err);
        exit(1);
    }
    // cl_kernel loss = clCreateKernel(prog, "RispSpikeLoss", &err);
    // if (err != CL_SUCCESS) {
    //     fprintf(stderr, "clCreateKernel for RispSpikeLoss failed: ");
    //     opencl_perror(err);
    //     exit(1);
    // }
    // cl_kernel bwd = clCreateKernel(prog, "RispDynamicsBwdKernel", &err);
    // if (err != CL_SUCCESS) {
    //     fprintf(stderr, "clCreateKernel for RispDynamicsBwdKernel failed: ");
    //     opencl_perror(err);
    //     exit(1);
    // }
    // cl_kernel update = clCreateKernel(prog, "RispWeightUpdates", &err);
    // if (err != CL_SUCCESS) {
    //     fprintf(stderr, "clCreateKernel for RispWeightUpdates failed: ");
    //     opencl_perror(err);
    //     exit(1);
    // }
    cl_kernel encode = clCreateKernel(prog, "RispEncodeInputs", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateKernel for RispEncodeInputs failed: ");
        opencl_perror(err);
        exit(1);
    }

    free(src);

    cl_mem x_buf = clCreateBuffer(
        ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        input_neurons * timesteps * sizeof(cl_float), NULL, &err);
    cl_mem thresh_buf =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       total_neurons * sizeof(cl_float), thresh, &err);
    cl_mem weights_buf = clCreateBuffer(
        ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        total_neurons * max_incoming * sizeof(cl_float), weights, &err);
    cl_mem delays_buf = clCreateBuffer(
        ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        total_neurons * max_incoming * sizeof(cl_int), delays, &err);
    cl_mem incoming_buf =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       total_neurons * sizeof(cl_int), incoming, &err);
    cl_mem incoming_ids_buf = clCreateBuffer(
        ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        total_neurons * max_incoming * sizeof(cl_int), incoming_ids, &err);
    cl_mem is_input_buf =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       total_neurons * sizeof(cl_int), is_input, &err);
    cl_mem v_buf =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                       input_neurons * sizeof(cl_float), NULL, &err);
    cl_mem s_buf =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                       input_neurons * timesteps * sizeof(cl_long), NULL, &err);
    cl_mem v_pre_buf = clCreateBuffer(
        ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        input_neurons * timesteps * sizeof(cl_float), NULL, &err);
    // cl_mem dL_ds_buf =
    //     clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
    //                    output_neurons * sizeof(cl_float), NULL, &err);
    // cl_mem spike_grad_history_buf = clCreateBuffer(
    //     ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
    //     total_neurons * timesteps * sizeof(cl_float), NULL, &err);
    // cl_mem voltage_grad_history_buf = clCreateBuffer(
    //     ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
    //     total_neurons * timesteps * sizeof(cl_float), NULL, &err);
    // cl_mem future_mem_grad_buf =
    //     clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
    //                    total_neurons * sizeof(cl_float), NULL, &err);
    // cl_mem delta_W_buf = clCreateBuffer(
    //     ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
    //     total_neurons * max_incoming * sizeof(cl_float), NULL, &err);
    // cl_mem m_weights_buf = clCreateBuffer(
    //     ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
    //     total_neurons * max_incoming * sizeof(cl_float), NULL, &err);
    // cl_mem v_weights_buf = clCreateBuffer(
    //     ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
    //     total_neurons * max_incoming * sizeof(cl_float), NULL, &err);
    cl_mem train_data_buf =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       train.observations * train.cols * 2 * sizeof(cl_float),
                       train.processed_data, &err);

    puts("Beginning training");
    double best_train_loss = DBL_MAX;
    double best_test_loss  = DBL_MAX;
    double best_train_acc  = 0.0;
    double best_test_acc   = 0.0;
    size_t* batch_order =
        (size_t*)calloc(train.observations, sizeof(*batch_order));
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss = 0.0;
        size_t correct    = 0;

        for (int i = 0; i < train.observations; i++) {
            batch_order[i] = i;
        }
        for (int i = 0; i < train.observations; i++) {
            size_t j       = rand() % train.observations;
            size_t tmp     = batch_order[i];
            batch_order[i] = batch_order[j];
            batch_order[j] = tmp;
        }

        // Batch processing loop
        for (int batch_start = 0; batch_start < train.observations;
             batch_start += batch_size) {
            Processor* p = nullptr;
            load_network(&p, n);

            double batch_loss    = 0.0;
            size_t batch_correct = 0;

            // Process batch samples
            for (int b = 0; (size_t)b < batch_size &&
                            (batch_start + b) < train.observations;
                 ++b) {
                size_t observation_idx = batch_order[batch_start + b];

                int* encoded_spikes  = (int*)calloc(input_neurons * timesteps,
                                                    sizeof(*encoded_spikes));
                EvaluationResults er = forward(&tb, p, &train, observation_idx,
                                               &nc, encoded_spikes);
                batch_loss += er.loss;
                batch_correct += er.correct;

                backward(&tb, &nc);

                // OpenCL impl
                {
                    size_t encode_size = input_neurons;
                    clSetKernelArg(encode, 0, sizeof(cl_mem), &x_buf);
                    clSetKernelArg(encode, 1, sizeof(cl_mem), &train_data_buf);
                    clSetKernelArg(encode, 2, sizeof(int), &train.cols);
                    clSetKernelArg(encode, 3, sizeof(int), &input_neurons);
                    clSetKernelArg(encode, 4, sizeof(int), &timesteps);
                    clSetKernelArg(encode, 5, sizeof(int), &observation_idx);
                    clSetKernelArg(encode, 6, sizeof(float),
                                   &spike_value_factor);

                    cl_int err = clEnqueueNDRangeKernel(queue, encode, 1, NULL,
                                                        &encode_size, NULL, 0,
                                                        NULL, NULL);
                    if (err != CL_SUCCESS) {
                        fprintf(stderr, "Encode kernel failed: %d ", err);
                        opencl_perror(err);
                        exit(1);
                    }

                    float* host_ptr = (float*)clEnqueueMapBuffer(
                        queue, x_buf, CL_TRUE, CL_MAP_READ, 0,
                        input_neurons * timesteps * sizeof(float), 0, NULL,
                        NULL, &err);

                    puts("Existing encoder");
                    for (int i = 0; i < input_neurons; i++) {
                        for (int t = 0; t < timesteps; t++) {
                            printf("%d", encoded_spikes[i * timesteps + t]);
                        }
                        puts("");
                    }
                    puts("OpenCL encoder");
                    for (int i = 0; i < input_neurons; i++) {
                        for (int t = 0; t < timesteps; t++) {
                            printf("%g", host_ptr[i * timesteps + t]);
                        }
                        puts("");
                    }
                    for (int i = 0; i < input_neurons; i++) {
                        for (int t = 0; t < timesteps; t++) {
                            if ((int)host_ptr[i * timesteps + t] !=
                                encoded_spikes[i * timesteps + t]) {
                                fprintf(stderr,
                                        "Encoding mismatch @ input=%d t=%d\n",
                                        i, t);
                                exit(1);
                            }
                        }
                    }

                    float f_val = 0.0f;
                    long l_val  = 0;
                    err         = clEnqueueFillBuffer(
                        queue, v_buf, &f_val, sizeof(float), 0,
                        total_neurons * sizeof(float), 0, NULL, NULL);
                    if (err != CL_SUCCESS) {
                        fprintf(stderr, "Fill v_buf failed: ");
                        opencl_perror(err);
                        exit(1);
                    }
                    err = clEnqueueFillBuffer(
                        queue, s_buf, &l_val, sizeof(long), 0,
                        total_neurons * timesteps * sizeof(long), 0, NULL,
                        NULL);
                    if (err != CL_SUCCESS) {
                        fprintf(stderr, "Fill s_buf failed: ");
                        opencl_perror(err);
                        exit(1);
                    }
                    err = clEnqueueFillBuffer(
                        queue, v_pre_buf, &f_val, sizeof(float), 0,
                        total_neurons * timesteps * sizeof(float), 0, NULL,
                        NULL);
                    if (err != CL_SUCCESS) {
                        fprintf(stderr, "Fill v_pre_buf failed: ");
                        opencl_perror(err);
                        exit(1);
                    }

                    size_t fwd_size = total_neurons;
                    clSetKernelArg(fwd, 0, sizeof(cl_mem), &x_buf);
                    clSetKernelArg(fwd, 1, sizeof(cl_mem), &thresh_buf);
                    clSetKernelArg(fwd, 2, sizeof(cl_mem), &weights_buf);
                    clSetKernelArg(fwd, 3, sizeof(cl_mem), &delays_buf);
                    clSetKernelArg(fwd, 4, sizeof(cl_mem), &incoming_buf);
                    clSetKernelArg(fwd, 5, sizeof(cl_mem), &incoming_ids_buf);
                    clSetKernelArg(fwd, 6, sizeof(float), &neuron_decay);
                    clSetKernelArg(fwd, 7, sizeof(float), &min_potential);
                    clSetKernelArg(fwd, 8, sizeof(cl_mem), &is_input_buf);
                    clSetKernelArg(fwd, 9, sizeof(cl_mem), &v_buf);
                    clSetKernelArg(fwd, 10, sizeof(cl_mem), &s_buf);
                    clSetKernelArg(fwd, 11, sizeof(cl_mem), &v_pre_buf);
                    clSetKernelArg(fwd, 12, sizeof(int), &total_neurons);
                    clSetKernelArg(fwd, 13, sizeof(int), &timesteps);
                    clSetKernelArg(fwd, 15, sizeof(int), &max_incoming);
                    for (int t = 0; t < timesteps; t++) {
                        clSetKernelArg(fwd, 14, sizeof(int), &t);

                        err = clEnqueueNDRangeKernel(queue, fwd, 1, NULL,
                                                     &fwd_size, NULL, 0, NULL,
                                                     NULL);
                        if (err != CL_SUCCESS) {
                            fprintf(stderr, "Fwd kernel failed @t=%d: ", t);
                            opencl_perror(err);
                            exit(1);
                        }
                    }

                    long* long_ptr = (long*)clEnqueueMapBuffer(
                        queue, s_buf, CL_TRUE, CL_MAP_READ, 0,
                        total_neurons * timesteps * sizeof(long), 0, NULL, NULL,
                        &err);

                    host_ptr = (float*)clEnqueueMapBuffer(
                        queue, v_pre_buf, CL_TRUE, CL_MAP_READ, 0,
                        total_neurons * timesteps * sizeof(long), 0, NULL, NULL,
                        &err);

                    puts("Existing v_pre:");
                    for (int i = 0; i < total_neurons; i++) {
                        for (int t = 0; t < timesteps; t++) {
                            printf("%g ", tb.v_pre[t][i]);
                        }
                        puts("");
                    }
                    puts("OpenCL v_pre:");
                    for (int i = 0; i < total_neurons; i++) {
                        for (int t = 0; t < timesteps; t++) {
                            printf("%g ", host_ptr[i * timesteps + t]);
                        }
                        puts("");
                    }

                    puts("Existing:");
                    for (int i = 0; i < total_neurons; i++) {
                        for (int t = 0; t < timesteps; t++) {
                            printf("%d", (int)tb.spikes[t][i]);
                        }
                        puts("");
                    }
                    puts("OpenCL:");
                    for (int i = 0; i < total_neurons; i++) {
                        for (int t = 0; t < timesteps; t++) {
                            printf("%d", (int)long_ptr[i * timesteps + t]);
                            if ((int)long_ptr[i * timesteps + t] !=
                                (int)tb.spikes[t][i]) {
                                printf("\nMismatch\n");
                                exit(1);
                            }
                        }
                        puts("");
                    }

                    exit(1);
                }
            }

            epoch_loss += batch_loss;
            correct += batch_correct;

            // Average gradients over the actual batch size.
            size_t current_batch_size = min(
                (size_t)batch_size, train.observations - (size_t)batch_start);
            weight_updates(&tb, &nc, &train, current_batch_size, batch_size,
                           batch_start, epoch);

            delete p;
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

        Processor* p = nullptr;
        load_network(&p, n);
        for (int i = 0; i < test.observations; i++) {
            p->clear_activity();

            encode_spikes(p, &test, i, timesteps, timeseries, input_neurons,
                          NULL);

            p->run(timesteps);
            const vector<int>& output_counts = p->output_counts();
            size_t max_idx                   = 0;
            int max_val                      = 0;
            for (int output = 0; output < output_neurons; output++) {
                tb.spike_logits[output] =
                    output_counts[output] / (double)timesteps;

                if (output_counts[output] > max_val) {
                    max_val = output_counts[output];
                    max_idx = output;
                }
            }

            softmax(tb.spike_logits.data(), tb.softmax_out.data(),
                    output_neurons);

            test_loss -= log(tb.softmax_out[(size_t)test.labels[i]]);
            test_correct += (max_idx == (size_t)test.labels[i]);
        }

        delete p;

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
        // printf("%g\n", best_test_loss);
    }

    delete n;
    free(train.data);
    free(train.labels);
    free(train.min_vals);
    free(train.max_vals);
    free(train.processed_data);
    free(test.data);
    free(test.labels);
    free(test.min_vals);
    free(test.max_vals);
    free(test.processed_data);
    free(batch_order);
}
