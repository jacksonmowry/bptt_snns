#include "csv.h"
#include "framework.hpp"
#include <cassert>
#include <cfloat>
#include <cstddef>
#include <fstream>

using namespace std;
using namespace neuro;
using nlohmann::json;

// Adam/Learning Parameters
#define LEARNING_RATE (1.0e-3)
#define C_DECAY (0.0000000001)
#define BETA1 (0.9)
#define BETA2 (0.999)
#define ADAM_EPS (1.0e-8)
#define EPOCHS (10000)
#define BATCH_SIZE (15)

// Network Topo/Size
#define INPUT_NEURONS (4)
#define HIDDEN_NEURONS (16)
#define OUTPUT_NEURONS (3)
#define TOTAL_NEURONS (INPUT_NEURONS + HIDDEN_NEURONS + OUTPUT_NEURONS)

// Dataset/Inference Parameters
#define TIMESTEPS (64)
#define NUM_FEATURES (4)

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
    double max = DBL_MIN;
    for (size_t i = 0; i < n; i++) {
        if (logits[i] > DBL_MIN) {
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

double spike_surrogate(double charge, double threshold) {
    double x = 1.0 + abs(charge - threshold);
    return 1.0 / (x * x);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s emptynet data.csv label.csv\n", argv[0]);
        return 1;
    }

    Dataset d = load_dataset(argv[2], argv[3]);

    srand(time(NULL));
    srand48(time(NULL));

    json emptynet;
    ifstream fin(argv[1]);
    fin >> emptynet;
    fin.close();

    Network* n = new Network();
    n->from_json(emptynet);

    string leak_prop = n->get_data("proc_params")["leak_mode"];
    bool leak        = leak_prop == "all";

    if (!n) {
        fprintf(stderr, "%s:%s:%d: Unable to create network.\n", __FILE__,
                __FUNCTION__, __LINE__);
        exit(1);
    }

    // 4 Input neurons, 4 Hidden neurons, 3 Output neurons
    const size_t layer_sizes[3]            = {INPUT_NEURONS, HIDDEN_NEURONS,
                                              OUTPUT_NEURONS};
    const size_t layer_cumulitive_sizes[3] = {0, INPUT_NEURONS,
                                              INPUT_NEURONS + HIDDEN_NEURONS};
    const size_t num_layers                = 3;
    size_t neuron_count                    = 0;

    for (size_t i = 0; i < num_layers; i++) {
        for (size_t j = 0; j < layer_sizes[i]; j++) {
            n->add_node(neuron_count)->set("Threshold", normal(1.0, 0.05));

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

    // Sparse recurrent connections within hidden layer (75% probability)
    for (size_t i = 0; i < TOTAL_NEURONS; i++) {
        for (size_t j = 0; j < TOTAL_NEURONS; j++) {
            // if (i == j)
            //     continue; // Skip self-loops
            if (drand48() < 0.75) {
                n->add_edge(i, j);

                double scale = 0.5 / max(1.0, sqrt(TOTAL_NEURONS));

                // 75% chance of being an excitatory synapse
                double rand_weight = normal(0.0, scale);
                n->get_edge(i, j)->set(n->get_edge_property("Weight")->index,
                                       rand_weight);
                n->get_edge(i, j)->set(n->get_edge_property("Delay")->index,
                                       rand() % 7 + 1);
            }
        }
    }

    // Encode Data/Labels
    puts("Encoding data");
    vector<vector<vector<bool>>> data_spikes(
        d.rows,
        vector<vector<bool>>(INPUT_NEURONS, vector<bool>(TIMESTEPS, false)));
    for (int i = 0; i < d.rows; i++) {
        for (size_t j = 0; j < INPUT_NEURONS; j++) {
            double x = (d.data[i * d.cols + j] - d.min_vals[j]) /
                       (d.max_vals[j] - d.min_vals[j]);
            x        = x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);

            for (double k = 0.0; k < (double)TIMESTEPS; k += 1.0 / x) {
                data_spikes[i][j][(size_t)k] = true;
            }
            // for (size_t k = 0; k < TIMESTEPS; k++) {
            //     if (drand48() < x) {
            //         data_spikes[i][j][k] = true;
            //     }
            // }
        }
    }

    double weights[TOTAL_NEURONS][TOTAL_NEURONS]   = {{0}};
    int delays[TOTAL_NEURONS][TOTAL_NEURONS]       = {{0}};
    double thresholds[TOTAL_NEURONS]               = {0};
    double m_weights[TOTAL_NEURONS][TOTAL_NEURONS] = {{0}};
    double v_weights[TOTAL_NEURONS][TOTAL_NEURONS] = {{0}};

    for (size_t i = 0; i < TOTAL_NEURONS; i++) {
        thresholds[i] = n->get_node(i)->get("Threshold");

        for (size_t j = 0; j < n->get_node(i)->outgoing.size(); j++) {
            Edge* e               = n->get_node(i)->outgoing[j];
            weights[i][e->to->id] = e->get("Weight");
            delays[i][e->to->id]  = e->get("Delay");
        }
    }

    int update_step = 0;

    puts("Beginning training");
    size_t* batch_order = (size_t*)calloc(d.rows, sizeof(*batch_order));
    for (size_t epoch = 0; epoch < EPOCHS; epoch++) {
        Processor* p = nullptr;
        load_network(&p, n);
        double epoch_loss = 0.0;
        size_t correct    = 0;
        size_t zero_fires = 0;
        double L2         = 0.0;

        for (int i = 0; i < d.rows; i++) {
            batch_order[i] = i;
        }
        for (int i = 0; i < d.rows; i++) {
            size_t j       = rand() % d.rows;
            size_t tmp     = batch_order[i];
            batch_order[i] = batch_order[j];
            batch_order[j] = tmp;
        }

        // Batch processing loop
        for (int batch_start = 0; batch_start < d.rows;
             batch_start += BATCH_SIZE) {
            double batch_loss                            = 0.0;
            size_t batch_correct                         = 0;
            double delta_W[TOTAL_NEURONS][TOTAL_NEURONS] = {{0}};

            // Process batch samples
            for (int b = 0; b < BATCH_SIZE && (batch_start + b) < d.rows; ++b) {
                size_t observation_idx = batch_order[batch_start + b];

                p->clear_activity();

                bool spikes[TOTAL_NEURONS][TIMESTEPS]    = {{0}};
                double charges[TOTAL_NEURONS][TIMESTEPS] = {{0.0}};
                double output_counts[OUTPUT_NEURONS]     = {0};

                // 1. Forward Pass
                for (size_t t = 0; t < TIMESTEPS; t++) {
                    for (size_t input = 0; input < INPUT_NEURONS; input++) {
                        p->apply_spike(
                            {(int)input, 0,
                             (double)data_spikes[observation_idx][input][t]});
                    }

                    p->run(1);

                    const vector<int>& neuron_counts     = p->neuron_counts();
                    const vector<double>& neuron_charges = p->neuron_charges();
                    for (size_t neuron = 0; neuron < TOTAL_NEURONS; neuron++) {
                        spikes[neuron][t] = neuron_counts[neuron];

                        // Neuron charges are reset to zero when they fire, so
                        // we lose their charge
                        if (neuron_counts[neuron]) {
                            charges[neuron][t] = thresholds[neuron];
                        } else {
                            charges[neuron][t] = neuron_charges[neuron];
                        }

                        // Keep track of output fire counts for decoding later
                        if (neuron >= layer_cumulitive_sizes[num_layers - 1]) {
                            output_counts[neuron -
                                          layer_cumulitive_sizes[num_layers -
                                                                 1]] +=
                                neuron_counts[neuron];
                        }
                    }
                }

                // Convert output_counts to logits
                size_t max_idx = 0;
                double max_val = 0;
                for (size_t neuron = 0; neuron < OUTPUT_NEURONS; neuron++) {
                    output_counts[neuron] /= (double)TIMESTEPS;

                    if (output_counts[neuron] > max_val) {
                        max_idx = neuron;
                        max_val = output_counts[neuron];
                    }
                }

                if (max_idx == (size_t)d.labels[observation_idx]) {
                    batch_correct++;
                }

                // Now we have logits, spikes, charges, and final charges
                // Beginning backward pass

                // 2. Backward pass
                // y_one_hot -> target
                // spikes -> spikes
                // pre_acts -> charges
                // _state -> last col of charges
                // logits -> output_counts

                double target[OUTPUT_NEURONS]             = {0};
                target[(size_t)d.labels[observation_idx]] = 1.0;

                double output_spike_gradient[OUTPUT_NEURONS] = {0.0};
                batch_loss +=
                    cross_entropy(output_counts, target, output_spike_gradient,
                                  OUTPUT_NEURONS);
                for (size_t i = 0; i < OUTPUT_NEURONS; i++) {
                    output_spike_gradient[i] /= (double)TIMESTEPS;
                }

                // grad_weights(python) -> delta_W[source][dest]
                double future_mem_grad[TOTAL_NEURONS]               = {0.0};
                double spike_grad_history[TOTAL_NEURONS][TIMESTEPS] = {{0.0}};

                for (int t = TIMESTEPS - 1; t >= 0; t--) {
                    double same_time_grad[TOTAL_NEURONS] = {0.0};
                    for (size_t neuron = 0; neuron < TOTAL_NEURONS; neuron++) {
                        same_time_grad[neuron] = spike_grad_history[neuron][t];
                    }

                    for (size_t output = 0; output < OUTPUT_NEURONS; output++) {
                        same_time_grad[layer_cumulitive_sizes[num_layers - 1] +
                                       output] += output_spike_gradient[output];
                    }

                    double next_future_mem_grad[TOTAL_NEURONS] = {0.0};

                    for (int dest = TOTAL_NEURONS - 1; dest >= 0; dest--) {
                        double s     = spikes[dest][t];
                        double u     = charges[dest][t];
                        double ds_du = spike_surrogate(u, thresholds[dest]);
                        double g_u   = future_mem_grad[dest] * (1.0 - s) +
                                       same_time_grad[dest] * ds_du;

                        for (size_t source = 0; source < TOTAL_NEURONS;
                             source++) {
                            if (weights[source][dest] == 0.0) {
                                continue;
                            }

                            int delay       = delays[source][dest];
                            int source_time = t - delay;
                            if (source_time < 0) {
                                continue;
                            }

                            double source_spike = spikes[source][source_time];
                            delta_W[source][dest] += source_spike * g_u;
                            spike_grad_history[source][source_time] +=
                                g_u * weights[source][dest];
                        }

                        next_future_mem_grad[dest] = alpha(leak) * g_u;
                    }

                    memcpy(future_mem_grad, next_future_mem_grad,
                           sizeof(future_mem_grad));
                }
            }

            // Average gradients over batch
            double inv_batch = 1.0 / BATCH_SIZE;
            for (size_t i = 0; i < TOTAL_NEURONS; i++) {
                for (size_t j = 0; j < TOTAL_NEURONS; j++) {
                    delta_W[i][j] *= inv_batch;

                    if (delta_W[i][j] >= 5.0) {
                        delta_W[i][j] = 5.0;
                    }
                    if (delta_W[i][j] <= -5.0) {
                        delta_W[i][j] = -5.0;
                    }
                }
            }

            epoch_loss += batch_loss;
            correct += batch_correct;

            // Adam update (moved outside sample loop)
            update_step++;
            double b1_t = 1.0 - pow(BETA1, update_step);
            double b2_t = 1.0 - pow(BETA2, update_step);
            for (size_t i = 0; i < TOTAL_NEURONS; i++) {
                for (size_t j = 0; j < n->get_node(i)->outgoing.size(); j++) {
                    Edge* e = n->get_node(i)->outgoing[j];
                    int k   = e->to->id;

                    m_weights[i][k] =
                        BETA1 * m_weights[i][k] + (1.0 - BETA1) * delta_W[i][k];
                    v_weights[i][k] =
                        BETA2 * v_weights[i][k] +
                        (1.0 - BETA2) * (delta_W[i][k] * delta_W[i][k]);

                    double mW_hat = m_weights[i][k] / b1_t;
                    double vW_hat = v_weights[i][k] / b2_t;

                    L2 += (LEARNING_RATE * mW_hat / (sqrt(vW_hat + ADAM_EPS))) *
                          (LEARNING_RATE * mW_hat / (sqrt(vW_hat + ADAM_EPS)));
                    weights[i][k] -=
                        LEARNING_RATE * mW_hat / (sqrt(vW_hat + ADAM_EPS));

                    e->set("Weight", weights[i][k]);
                }
            }
        }

        printf("Epoch: %zu/%zu, Loss: %f, Acc: %f, Zero Fires: %zu, L2: %g\n",
               epoch + 1, (size_t)EPOCHS, epoch_loss / (double)d.rows,
               correct / (double)d.rows, zero_fires, L2);
        delete p;
    }

    delete n;
    free(d.data);
    free(d.labels);
    free(d.min_vals);
    free(d.max_vals);
    free(batch_order);
}
