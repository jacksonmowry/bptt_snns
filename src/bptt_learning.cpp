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
    fprintf(stderr, "  -h, --help                     Show this help\n");
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
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0},
    };

    int c;
    char* endptr;

    while ((c = getopt_long(argc, argv, "n:d:l:b:c:r:e:u:o:t:H:s:hp:B:",
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
        load_dataset_2d(data_file, label_file, 0.8, &train, &test);
    } else {
        load_dataset(data_file, label_file, 0.8, &train, &test);
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

    // 4 Input neurons, 4 Hidden neurons, 3 Output neurons
    const size_t layer_sizes[3]            = {input_neurons, hidden_neurons,
                                              output_neurons};
    const size_t layer_cumulitive_sizes[3] = {0, input_neurons,
                                              input_neurons + hidden_neurons};
    const size_t num_layers                = 3;
    size_t neuron_count                    = 0;
    size_t synapse_count                   = 0;

    for (size_t i = 0; i < num_layers; i++) {
        for (size_t j = 0; j < layer_sizes[i]; j++) {
            n->add_node(neuron_count)->set("Threshold", 1.0);

            if (i == 0) {
                // Input
                n->add_input(neuron_count);
            } else if (i == num_layers - 1) {
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

    vector<vector<double>> weights(total_neurons);
    vector<vector<double>> delta_W(total_neurons);
    vector<vector<int>> delays(total_neurons);
    vector<double> thresholds(total_neurons);
    vector<vector<double>> m_weights(total_neurons);
    vector<vector<double>> v_weights(total_neurons);
    vector<vector<double>> spikes(timesteps, vector<double>(total_neurons));
    vector<vector<double>> v_pre(timesteps, vector<double>(total_neurons));
    vector<double> spike_logits(output_neurons);
    vector<double> target(output_neurons);
    vector<double> dL_ds(output_neurons);
    vector<double> future_mem_grad(total_neurons);
    vector<vector<double>> spike_grad_history(total_neurons,
                                              vector<double>(timesteps));
    vector<vector<double>> voltage_grad_history(total_neurons,
                                                vector<double>(timesteps));
    vector<double> softmax_out(output_neurons);

    Eigen::VectorXd future_mem_grad_(total_neurons);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> sgh(
        total_neurons, timesteps);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> vgh(
        total_neurons, timesteps);
    Eigen::VectorXd dL_dV_(total_neurons);
    Eigen::VectorXd v_pre_t_(total_neurons);
    Eigen::VectorXd dV_post_dV_pre_(total_neurons);
    Eigen::VectorXd dV_post_ds_t_(total_neurons);
    Eigen::VectorXd ds_t_dV_pre_(total_neurons);
    Eigen::VectorXd dV_leak_dV_t1_(total_neurons);
    Eigen::VectorXd grad_(total_neurons);

    double b1_t = 1.0;
    double b2_t = 1.0;

    for (size_t i = 0; i < total_neurons; i++) {
        thresholds[i] = n->get_node(i)->get("Threshold");
        weights[i].reserve(n->get_node(i)->incoming.size());
        delays[i].reserve(n->get_node(i)->incoming.size());
        delta_W[i].resize(n->get_node(i)->incoming.size());
        m_weights[i].resize(n->get_node(i)->incoming.size());
        v_weights[i].resize(n->get_node(i)->incoming.size());

        for (size_t j = 0; j < n->get_node(i)->incoming.size(); j++) {
            Edge* e = n->get_node(i)->incoming[j];

            weights[i].push_back(e->get("Weight"));
            delays[i].push_back(e->get("Delay"));
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
                // size_t observation_idx = b;
                size_t observation_idx = batch_order[batch_start + b];

                p->clear_activity();

                for (size_t i = 0; i < timesteps; i++) {
                    fill(spikes[i].begin(), spikes[i].end(), 0.0);
                    fill(v_pre[i].begin(), v_pre[i].end(), 0.0);
                }
                fill(spike_logits.begin(), spike_logits.end(), 0.0);

                encode_spikes(p, &train, observation_idx, timesteps, timeseries,
                              input_neurons);

                // 1. Forward Pass
                for (size_t t = 0; t < timesteps; t++) {
                    p->run(1);

                    const vector<int>& neuron_counts = p->neuron_counts();
                    const vector<double>& neuron_pre_charges =
                        p->neuron_pre_charges();
                    for (size_t neuron = 0; neuron < total_neurons; neuron++) {
                        spikes[t][neuron] = neuron_counts[neuron];
                        v_pre[t][neuron]  = neuron_pre_charges[neuron];

                        // Keep track of output fire counts for decoding later
                        if (neuron >= layer_cumulitive_sizes[num_layers - 1]) {
                            spike_logits[neuron -
                                         layer_cumulitive_sizes[num_layers -
                                                                1]] +=
                                neuron_counts[neuron];
                        }
                    }
                }

                // Convert output_counts to logits
                size_t max_idx = 0;
                double max_val = 0;
                for (size_t neuron = 0; neuron < output_neurons; neuron++) {
                    spike_logits[neuron] /= (double)timesteps;

                    if (spike_logits[neuron] > max_val) {
                        max_idx = neuron;
                        max_val = spike_logits[neuron];
                    }
                }

                if (max_idx == (size_t)train.labels[observation_idx]) {
                    batch_correct++;
                }

                // 2. Backward pass
                for (size_t i = 0; i < output_neurons; i++) {
                    if (i == (size_t)train.labels[observation_idx]) {
                        target[i] = 1.00f;
                    } else {
                        target[i] = 0.00f;
                    }
                }

                double loss_spike =
                    cross_entropy(spike_logits.data(), target.data(),
                                  dL_ds.data(), output_neurons);

                batch_loss += loss_spike;

                fill(future_mem_grad.begin(), future_mem_grad.end(), 0.0);
                for (size_t i = 0; i < total_neurons; i++) {
                    fill(spike_grad_history[i].begin(),
                         spike_grad_history[i].end(), 0.0);
                    fill(voltage_grad_history[i].begin(),
                         voltage_grad_history[i].end(), 0.0);
                }

                future_mem_grad_.setZero();
                sgh.setZero();
                vgh.setZero();
                dL_dV_.setZero();
                v_pre_t_.setZero();
                dV_post_dV_pre_.setZero();
                dV_post_ds_t_.setZero();
                ds_t_dV_pre_.setZero();
                dV_leak_dV_t1_.setZero();
                grad_.setZero();

                for (int t = timesteps - 1; t >= 0; t--) {
                    sgh.col(t).segment(layer_cumulitive_sizes[2],
                                       output_neurons) +=
                        Eigen::Map<const Eigen::VectorXd>(&dL_ds[0],
                                                          output_neurons) /
                        timesteps;

                    // Old for loop
                    for (size_t output = 0; output < output_neurons; output++) {
                        spike_grad_history[layer_cumulitive_sizes[num_layers -
                                                                  1] +
                                           output][t] +=
                            (dL_ds[output] / timesteps);
                    }

                    dL_dV_ = vgh.col(t);
                    dL_dV_ += future_mem_grad_;
                    v_pre_t_ = Eigen::Map<const Eigen::VectorXd>(&v_pre[t][0],
                                                                 total_neurons);

                    dV_post_dV_pre_ = (Eigen::Map<const Eigen::VectorXd>(
                                           &spikes[t][0], total_neurons)
                                           .array() <= 0)
                                          .cast<double>();

                    dV_post_ds_t_ = -v_pre_t_;
                    if (min_potential > 0) {
                        (dV_post_ds_t_.array() + min_potential).matrix();
                    }

                    ds_t_dV_pre_ =
                        (rho / (2.0 * tau)) *
                        (-(v_pre_t_ - Eigen::Map<const Eigen::VectorXd>(
                                          &thresholds[0], total_neurons))
                              .array()
                              .abs()
                              .matrix() /
                         tau)
                            .array()
                            .exp()
                            .matrix();

                    dV_leak_dV_t1_ =
                        (v_pre_t_.array() >= min_potential).cast<double>() *
                        (1.0 - leak);

                    grad_ = (dL_dV_.array() * dV_post_dV_pre_.array()) +
                            (dL_dV_.array() * dV_post_ds_t_.array() *
                             ds_t_dV_pre_.array()) +
                            (sgh.col(t).array() * ds_t_dV_pre_.array());

                    future_mem_grad_ =
                        (dL_dV_.array() * dV_post_dV_pre_.array() *
                         dV_leak_dV_t1_.array()) +
                        (dL_dV_.array() * dV_post_ds_t_.array() *
                         ds_t_dV_pre_.array() * dV_leak_dV_t1_.array()) +
                        (sgh.col(t).array() * ds_t_dV_pre_.array() *
                         dV_leak_dV_t1_.array());

                    for (int dest = total_neurons - 1; dest >= 0; dest--) {
                        for (size_t source_idx = 0;
                             source_idx < n->get_node(dest)->incoming.size();
                             source_idx++) {
                            size_t source = n->get_node(dest)
                                                ->incoming[source_idx]
                                                ->from->id;

                            int delay       = delays[dest][source_idx];
                            int source_time = t - delay;
                            if (source_time < 0) {
                                continue;
                            }

                            double source_spike = spikes[source_time][source];
                            delta_W[dest][source_idx] +=
                                source_spike * grad_(dest);
                            sgh(source, source_time) +=
                                grad_(dest) * weights[dest][source_idx];
                        }
                    }
                }
            }

            // Average gradients over the actual batch size.
            size_t current_batch_size = min(
                (size_t)batch_size, train.observations - (size_t)batch_start);
            double inv_batch = 1.0 / ((double)current_batch_size * timesteps);

            epoch_loss += batch_loss;
            correct += batch_correct;

            // Adam update
            b1_t *= BETA1;
            b2_t *= BETA2;

            for (size_t i = 0; i < total_neurons; i++) {
                for (size_t j = 0; j < n->get_node(i)->incoming.size(); j++) {
                    Edge* e = n->get_node(i)->incoming[j];

                    delta_W[i][j] *= inv_batch;

                    m_weights[i][j] =
                        BETA1 * m_weights[i][j] + (1.0 - BETA1) * delta_W[i][j];
                    v_weights[i][j] =
                        BETA2 * v_weights[i][j] +
                        (1.0 - BETA2) * (delta_W[i][j] * delta_W[i][j]);
                    delta_W[i][j] = 0.0;

                    double mW_hat = m_weights[i][j] / (1.0 - b1_t);
                    double vW_hat = v_weights[i][j] / (1.0 - b2_t);

                    double lr = learning_rate;
                    if (epoch == 0) {
                        lr = ((batch_start + batch_size) /
                              (double)train.observations) *
                             learning_rate;
                    }

                    weights[i][j] -= lr * mW_hat / (sqrt(vW_hat + ADAM_EPS));
                    weights[i][j] -= lr * decay_rate * weights[i][j];
                    e->set("Weight", weights[i][j]);
                }
            }

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

            encode_spikes(p, &test, i, timesteps, timeseries, input_neurons);

            p->run(timesteps);
            const vector<int>& output_counts = p->output_counts();
            size_t max_idx                   = 0;
            int max_val                      = 0;
            for (size_t output = 0; output < output_neurons; output++) {
                spike_logits[output] =
                    output_counts[output] / (double)timesteps;

                if (output_counts[output] > max_val) {
                    max_val = output_counts[output];
                    max_idx = output;
                }
            }

            softmax(spike_logits.data(), softmax_out.data(), output_neurons);

            test_loss -= log(softmax_out[(size_t)test.labels[i]]);
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
    free(test.data);
    free(test.labels);
    free(test.min_vals);
    free(test.max_vals);
    free(batch_order);
}
