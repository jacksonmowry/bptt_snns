#include "csv.h"
#include "framework.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstddef>
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

    size_t input_neurons;
    size_t hidden_neurons;
    size_t output_neurons;
    size_t layer_offsets[3];
    size_t total_neurons;

    size_t timesteps;
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

void encode_spikes(Processor* p, const Dataset* d, size_t index,
                   size_t timesteps, bool timeseries, size_t input_neurons) {
    if (timeseries) {
        size_t encoding_window = timesteps / d->cols;
        assert(encoding_window > 0);

        for (size_t input = 0; input < input_neurons / 2; input++) {
            for (int column_t = 0; column_t < d->cols; column_t++) {
                double encoding_start = column_t * encoding_window;
                double encoding_end   = encoding_start + encoding_window;

                double x =
                    (d->data[(index * d->rows_per_observation * d->cols) +
                             (input * d->cols) + column_t] -
                     d->min_vals[input]) /
                    (d->max_vals[input] - d->min_vals[input]);
                double inv_x = 1.0 - x;

                if (x > 0.0) {
                    for (double j = encoding_start; j < encoding_end;
                         j += 1.0 / x) {
                        p->apply_spike({(int)input * 2, (double)(int)j, 1.0});
                    }
                }
                if (inv_x > 0.0) {
                    for (double j = encoding_start; j < encoding_end;
                         j += 1.0 / inv_x) {
                        p->apply_spike(
                            {(int)input * 2 + 1, (double)(int)j, 1.0});
                    }
                }
            }
        }
    } else {
        for (size_t input = 0; input < input_neurons / 2; input++) {
            double x = (d->data[index * d->cols + input] - d->min_vals[input]) /
                       (d->max_vals[input] - d->min_vals[input]);
            double inv_x = 1.0 - x;
            if (x > 0.0) {
                for (double j = 0; j < (double)timesteps; j += 1.0 / x) {
                    p->apply_spike({(int)input * 2, (double)(size_t)j, 1.0});
                }
            }
            if (inv_x > 0.0) {
                for (double j = 0; j < (double)timesteps; j += 1.0 / inv_x) {
                    p->apply_spike(
                        {(int)input * 2 + 1, (double)(size_t)j, 1.0});
                }
            }
        }
    }
}

EvaluationResults forward(TrainingBundle* tb, Processor* p, const Dataset* d,
                          size_t index, const NetworkConfiguration* nc) {
    EvaluationResults er = {0.0, 0.0};

    p->clear_activity();

    for (size_t t = 0; t < nc->timesteps; t++) {
        fill(tb->spikes[t].begin(), tb->spikes[t].end(), 0.0);
        fill(tb->v_pre[t].begin(), tb->v_pre[t].end(), 0.0);
    }
    fill(tb->spike_logits.begin(), tb->spike_logits.end(), 0.0);

    encode_spikes(p, d, index, nc->timesteps, nc->timeseries,
                  nc->input_neurons);

    for (size_t t = 0; t < nc->timesteps; t++) {
        p->run(1);

        const vector<int>& neuron_counts         = p->neuron_counts();
        const vector<double>& neuron_pre_charges = p->neuron_pre_charges();
        for (size_t neuron = 0; neuron < nc->total_neurons; neuron++) {
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
    for (size_t neuron = 0; neuron < nc->output_neurons; neuron++) {
        tb->spike_logits[neuron] /= (double)nc->timesteps;

        if (tb->spike_logits[neuron] > max_val) {
            max_idx = neuron;
            max_val = tb->spike_logits[neuron];
        }
    }

    if (max_idx == (size_t)d->labels[index]) {
        er.correct++;
    }

    for (size_t i = 0; i < nc->output_neurons; i++) {
        if (i == (size_t)d->labels[index]) {
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

    for (size_t i = 0; i < nc->total_neurons; i++) {
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
    fprintf(stderr,
            "  -N, --network_json_out FILE    Network json out filename\n");
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
    size_t timesteps        = 32;
    size_t hidden_neurons   = 16;
    unsigned long seed      = (unsigned long)time(NULL);
    size_t epochs           = 10;
    size_t batch_size       = 1;
    double training_percent = 0.8;
    char* network_json_out  = NULL;

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
        {"network_json_out", required_argument, 0, 'N'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0},
    };

    int c;
    char* endptr;

    while ((c = getopt_long(argc, argv, "n:d:l:b:c:r:e:u:o:t:H:s:hp:B:P:N:",
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
        case 'N':
            network_json_out = optarg;
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

    string leak_prop     = n->get_data("proc_params")["leak_mode"];
    bool leak            = leak_prop == "all";
    double min_potential = n->get_data("proc_params")["min_potential"];

    if (!n) {
        fprintf(stderr, "%s:%s:%d: Unable to create network.\n", __FILE__,
                __FUNCTION__, __LINE__);
        exit(1);
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
    };

    size_t neuron_count  = 0;
    size_t synapse_count = 0;

    for (size_t i = 0; i < NUM_LAYERS; i++) {
        for (size_t j = 0; j < layer_sizes[i]; j++) {
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

    for (size_t i = 0; i < total_neurons; i++) {
        for (size_t j = 0; j < total_neurons; j++) {
            if (drand48() < (1.0 - connectivity)) {
                continue;
            }

            Edge* e = n->add_edge(i, j);
            synapse_count++;

            e->set(n->get_edge_property("Weight")->index, normal(0.0, 0.1));
            e->set(n->get_edge_property("Delay")->index, rand() % 7 + 1);
        }
    }

    printf("Neurons: %zu, Synapses: %zu\n", neuron_count, synapse_count);

    TrainingBundle tb(total_neurons, timesteps, output_neurons, tau, rho,
                      learning_rate, decay_rate);

    for (size_t i = 0; i < total_neurons; i++) {
        tb.thresholds[i] = n->get_node(i)->get("Threshold");
        tb.weights[i].reserve(n->get_node(i)->incoming.size());
        tb.delays[i].reserve(n->get_node(i)->incoming.size());
        tb.delta_W[i].resize(n->get_node(i)->incoming.size());
        tb.m_weights[i].resize(n->get_node(i)->incoming.size());
        tb.v_weights[i].resize(n->get_node(i)->incoming.size());

        for (size_t j = 0; j < n->get_node(i)->incoming.size(); j++) {
            Edge* e = n->get_node(i)->incoming[j];

            tb.weights[i].push_back(e->get("Weight"));
            tb.delays[i].push_back(e->get("Delay"));
        }
    }

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

                EvaluationResults er =
                    forward(&tb, p, &train, observation_idx, &nc);
                batch_loss += er.loss;
                batch_correct += er.correct;

                backward(&tb, &nc);
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
        for (int i = 0; i < train.observations; i++) {
            p->clear_activity();

            encode_spikes(p, &train, i, timesteps, timeseries, input_neurons);

            p->run(timesteps);
            const vector<int>& output_counts = p->output_counts();
            size_t max_idx                   = 0;
            int max_val                      = 0;
            for (size_t output = 0; output < output_neurons; output++) {
                tb.spike_logits[output] =
                    output_counts[output] / (double)timesteps;

                if (output_counts[output] > max_val) {
                    max_val = output_counts[output];
                    max_idx = output;
                }
            }

            softmax(tb.spike_logits.data(), tb.softmax_out.data(),
                    output_neurons);

            test_loss -= log(tb.softmax_out[(size_t)train.labels[i]]);
            test_correct += (max_idx == (size_t)train.labels[i]);
        }

        delete p;

        if (train.observations > 0) {
            test_correct /= train.observations;
            test_loss /= train.observations;

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

        if (best_test_acc == 1.0) {
            return 0;
        }
        // printf("%g\n", best_test_loss);
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
}
