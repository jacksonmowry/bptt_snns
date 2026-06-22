// 0.8989136 and parameters: {'learning_rate': 0.008407719088248392,
// 'decay_rate': 0.0007030429010269196, 'tau': 0.4784328025223471,
// 'rho': 1.206692630116519}. Best is trial 32 with value: 0.8989136.
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
// #define LEARNING_RATE (8.0e-3)
// #define C_DECAY (0.0000001)
#define BETA1 (0.9)
#define BETA2 (0.999)
#define ADAM_EPS (1.0e-8)
#define EPOCHS (1000)
#define BATCH_SIZE (10)
#define SCALE_RHO (1.0)
#define TAU_RHO_SCALED (0.9)
#define W (1)

// Network Topo/Size
#define INPUT_NEURONS (8)
#define HIDDEN_NEURONS (8)
#define OUTPUT_NEURONS (3)
#define TOTAL_NEURONS (INPUT_NEURONS + HIDDEN_NEURONS + OUTPUT_NEURONS)

// Dataset/Inference Parameters
#define TIMESTEPS (32)
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
                       double tau_rho_scaled, double w) {
    return (scale_rho / (2.0f * tau_rho_scaled)) *
           expf(-fabs(w * (v_pre_t - v_thresh)) / tau_rho_scaled);
}

int main(int argc, char* argv[]) {
    if (argc != 8) {
        fprintf(stderr,
                "usage: %s emptynet data.csv label.csv lr decay tau rho\n",
                argv[0]);
        return 1;
    }

    // srand(42);
    // srand48(42);
    srand(time(NULL));
    srand48(time(NULL));

    Dataset train;
    Dataset test;
    load_dataset(argv[2], argv[3], 0.8, &train, &test);
    double learning_rate = strtod(argv[4], NULL);
    double decay_rate    = strtod(argv[5], NULL);
    double tau           = strtod(argv[6], NULL);
    double rho           = strtod(argv[7], NULL);

    json emptynet;
    ifstream fin(argv[1]);
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
    const size_t layer_sizes[3]            = {INPUT_NEURONS, HIDDEN_NEURONS,
                                              OUTPUT_NEURONS};
    const size_t layer_cumulitive_sizes[3] = {0, INPUT_NEURONS,
                                              INPUT_NEURONS + HIDDEN_NEURONS};
    const size_t num_layers                = 3;
    size_t neuron_count                    = 0;

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

    for (size_t i = 1; i < num_layers; i++) {
        for (size_t source = layer_cumulitive_sizes[i - 1];
             source < layer_cumulitive_sizes[i]; source++) {
            for (size_t dest = layer_cumulitive_sizes[i];
                 dest < layer_cumulitive_sizes[i] + layer_sizes[i]; dest++) {
                Edge* e = n->add_edge(source, dest);

                e->set(n->get_edge_property("Weight")->index, normal(0.0, 0.1));
                e->set(n->get_edge_property("Delay")->index, rand() % 7 + 1);
            }
        }
    }

    // Input to Hidden
    // size_t ih = 0;
    // for (size_t source = 0; source < INPUT_NEURONS; source++) {
    //     for (size_t dest = 0; dest < HIDDEN_NEURONS; dest++) {
    //         Edge* e = n->add_edge(source, dest + INPUT_NEURONS);
    //         ih++;

    //         e->set("Delay", rand() % 7 + 1);
    //         e->set("Weight", normal(0, sqrt(INPUT_NEURONS)));
    //     }
    // }
    // printf("Added %zu ih connections\n", ih);

    // // Hidden to Hidden w/o self loops
    // size_t hh = 0;
    // for (size_t source = 0; source < HIDDEN_NEURONS; source++) {
    //     for (size_t dest = 0; dest < HIDDEN_NEURONS; dest++) {
    //         if (source == dest || drand48() <= 0.0) {
    //             continue;
    //         }

    //         Edge* e = n->add_edge(source + INPUT_NEURONS, dest +
    //         INPUT_NEURONS); hh++;

    //         e->set("Delay", rand() % 7 + 1);
    //         e->set("Weight", normal(0, sqrt(HIDDEN_NEURONS - 1)));
    //     }
    // }
    // printf("Added %zu hh connections\n", hh);

    // // Hidden to Output
    // size_t ho = 0;
    // for (size_t source = 0; source < HIDDEN_NEURONS; source++) {
    //     for (size_t dest = 0; dest < OUTPUT_NEURONS; dest++) {
    //         Edge* e = n->add_edge(source + INPUT_NEURONS,
    //                               dest + INPUT_NEURONS + HIDDEN_NEURONS);
    //         ho++;

    //         e->set("Delay", rand() % 7 + 1);
    //         e->set("Weight", normal(0, sqrt(HIDDEN_NEURONS)));
    //     }
    // }
    // printf("Added %zu ho connections\n", ho);

    // Encode Data/Labels
    puts("Encoding data");
    vector<vector<vector<bool>>> train_spikes(
        train.rows,
        vector<vector<bool>>(INPUT_NEURONS, vector<bool>(TIMESTEPS, false)));
    for (int i = 0; i < train.rows; i++) {
        for (size_t j = 0; j < INPUT_NEURONS / 2; j++) {
            double x = (train.data[i * train.cols + j] - train.min_vals[j]) /
                       (train.max_vals[j] - train.min_vals[j]);
            x        = x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
            double inv_x = 1.0 - x;

            if (x > 0.0) {
                for (double k = 0.0; k < (double)TIMESTEPS; k += 1.0 / x) {
                    train_spikes[i][j * 2][(size_t)k] = true;
                }
            }
            if (inv_x > 0.0) {
                for (double k = 0.0; k < (double)TIMESTEPS; k += 1.0 / inv_x) {
                    train_spikes[i][j * 2 + 1][(size_t)k] = true;
                }
            }
        }
    }

    vector<vector<vector<bool>>> test_spikes(
        test.rows,
        vector<vector<bool>>(INPUT_NEURONS, vector<bool>(TIMESTEPS, false)));
    for (int i = 0; i < test.rows; i++) {
        for (size_t j = 0; j < INPUT_NEURONS / 2; j++) {
            double x     = (test.data[i * test.cols + j] - test.min_vals[j]) /
                           (test.max_vals[j] - test.min_vals[j]);
            x            = x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
            double inv_x = 1.0 - x;

            if (x > 0.0) {
                for (double k = 0.0; k < (double)TIMESTEPS; k += 1.0 / x) {
                    test_spikes[i][j * 2][(size_t)k] = true;
                }
            }
            if (inv_x > 0.0) {
                for (double k = 0.0; k < (double)TIMESTEPS; k += 1.0 / inv_x) {
                    test_spikes[i][j * 2 + 1][(size_t)k] = true;
                }
            }
        }
    }

    double weights[TOTAL_NEURONS][TOTAL_NEURONS]   = {{0}};
    int delays[TOTAL_NEURONS][TOTAL_NEURONS]       = {{0}};
    double thresholds[TOTAL_NEURONS]               = {0};
    double m_weights[TOTAL_NEURONS][TOTAL_NEURONS] = {{0}};
    double v_weights[TOTAL_NEURONS][TOTAL_NEURONS] = {{0}};
    double b1_t                                    = 1.0;
    double b2_t                                    = 1.0;

    for (size_t i = 0; i < TOTAL_NEURONS; i++) {
        thresholds[i] = n->get_node(i)->get("Threshold");

        for (size_t j = 0; j < n->get_node(i)->outgoing.size(); j++) {
            Edge* e               = n->get_node(i)->outgoing[j];
            weights[i][e->to->id] = e->get("Weight");
            delays[i][e->to->id]  = e->get("Delay");
        }
    }

    puts("Beginning training");
    double best_train_loss = DBL_MAX;
    double best_test_loss  = DBL_MAX;
    double best_train_acc  = 0.0;
    double best_test_acc   = 0.0;
    size_t* batch_order    = (size_t*)calloc(train.rows, sizeof(*batch_order));
    for (size_t epoch = 0; epoch < EPOCHS; epoch++) {
        double epoch_loss = 0.0;
        size_t correct    = 0;
        size_t zero_fires = 0;
        double L2         = 0.0;

        for (int i = 0; i < train.rows; i++) {
            batch_order[i] = i;
        }
        for (int i = 0; i < train.rows; i++) {
            size_t j       = rand() % train.rows;
            size_t tmp     = batch_order[i];
            batch_order[i] = batch_order[j];
            batch_order[j] = tmp;
        }

        // Batch processing loop
        for (int batch_start = 0; batch_start < train.rows;
             batch_start += BATCH_SIZE) {
            Processor* p = nullptr;
            load_network(&p, n);

            double batch_loss                            = 0.0;
            size_t batch_correct                         = 0;
            double delta_W[TOTAL_NEURONS][TOTAL_NEURONS] = {{0}};

            // Process batch samples
            for (int b = 0; b < BATCH_SIZE && (batch_start + b) < train.rows;
                 ++b) {
                // size_t observation_idx = b;
                size_t observation_idx = batch_order[batch_start + b];

                p->clear_activity();

                bool spikes[TOTAL_NEURONS][TIMESTEPS]  = {{0}};
                double v_pre[TOTAL_NEURONS][TIMESTEPS] = {{0.0}};
                double spike_logits[OUTPUT_NEURONS]    = {0};

                // 1. Forward Pass
                for (size_t t = 0; t < TIMESTEPS; t++) {
                    for (size_t input = 0; input < INPUT_NEURONS; input++) {
                        p->apply_spike(
                            {(int)input, 0,
                             (double)train_spikes[observation_idx][input][t]});
                    }

                    p->run(1);

                    const vector<int>& neuron_counts     = p->neuron_counts();
                    const vector<double>& neuron_charges = p->neuron_charges();
                    const vector<double>& neuron_pre_charges =
                        p->neuron_pre_charges();
                    for (size_t neuron = 0; neuron < TOTAL_NEURONS; neuron++) {
                        spikes[neuron][t] = neuron_counts[neuron];
                        v_pre[neuron][t]  = neuron_pre_charges[neuron];

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
                for (size_t neuron = 0; neuron < OUTPUT_NEURONS; neuron++) {
                    spike_logits[neuron] /= (double)TIMESTEPS;

                    if (spike_logits[neuron] > max_val) {
                        max_idx = neuron;
                        max_val = spike_logits[neuron];
                    }
                }

                if (max_idx == (size_t)train.labels[observation_idx]) {
                    batch_correct++;
                }

                // 2. Backward pass
                double target[OUTPUT_NEURONS] = {0};
                for (size_t i = 0; i < OUTPUT_NEURONS; i++) {
                    if (i == (size_t)train.labels[observation_idx]) {
                        target[i] = 1;
                    } else {
                        target[i] = 0.00;
                    }
                }
                double dL_ds[OUTPUT_NEURONS] = {0.0};
                double loss_spike =
                    cross_entropy(spike_logits, target, dL_ds, OUTPUT_NEURONS);

                // double voltage_logits[OUTPUT_NEURONS] = {0.0};
                // for (size_t n = 0; n < OUTPUT_NEURONS; n++) {
                //     for (size_t t = 0; t < TIMESTEPS; t++) {
                //         voltage_logits[n] += v_pre[n][t];
                //     }
                //     voltage_logits[n] /= TIMESTEPS;
                // }
                // double dL_dy[OUTPUT_NEURONS] = {0.0};
                // double loss_voltage = cross_entropy(voltage_logits, target,
                //                                     dL_dy, OUTPUT_NEURONS);
                // dL_dy[0]            = 0;
                // dL_dy[1]            = 0;

                batch_loss += loss_spike; // + 0.3 * loss_voltage;

                double future_mem_grad[TOTAL_NEURONS]                 = {0.0};
                double spike_grad_history[TOTAL_NEURONS][TIMESTEPS]   = {{0.0}};
                double voltage_grad_history[TOTAL_NEURONS][TIMESTEPS] = {{0.0}};

                // printf("[");
                // for (size_t i = 0; i < OUTPUT_NEURONS; i++) {
                //     printf("%g (%g)", dL_ds[i], dL_ds[i] + target[i]);
                // }
                // printf("], target: %zu", (size_t)d.labels[observation_idx]);
                // puts("");

                for (int t = TIMESTEPS - 1; t >= 0; t--) {
                    for (size_t output = 0; output < OUTPUT_NEURONS; output++) {
                        // voltage_grad_history[layer_cumulitive_sizes[num_layers
                        // -
                        //                                             1] +
                        //                      output][t] +=
                        //     (dL_dy[output] / TIMESTEPS);
                        spike_grad_history[layer_cumulitive_sizes[num_layers -
                                                                  1] +
                                           output][t] +=
                            (dL_ds[output] / TIMESTEPS);
                        // (dL_ds[output] / TIMESTEPS);
                    }

                    // Y is the output of a neuron
                    // X is the input to that neuron

                    double next_future_mem_grad[TOTAL_NEURONS] = {0.0};

                    for (int dest = TOTAL_NEURONS - 1; dest >= INPUT_NEURONS;
                         dest--) {
                        // Start Slayer copy
                        double dL_dV = voltage_grad_history[dest][t] / W;
                        dL_dV += future_mem_grad[dest];

                        double v_pre_t = v_pre[dest][t];

                        double dV_post_dV_pre =
                            1.0 - (spikes[dest][t] > 0 ? 1.0 : 0.0);
                        double dV_pre_dx_t = W;
                        double dV_post_ds_t =
                            (min_potential * W > 0)
                                ? ((min_potential - v_pre_t) * W)
                                : (-v_pre_t * W);
                        double ds_t_dV_pre = spike_surrogate(
                            v_pre_t, thresholds[dest], rho, tau, 1);

                        double dV_pre_dV_leak = 1.0;
                        double dV_leak_dV_t1 =
                            (v_pre_t * W >= min_potential * W) ? (1.0 - leak)
                                                               : 0.0;

                        double grad =
                            (dL_dV * dV_post_dV_pre * dV_pre_dx_t) +
                            (dL_dV * dV_post_ds_t * ds_t_dV_pre * dV_pre_dx_t) +
                            ((spike_grad_history[dest][t] * ds_t_dV_pre *
                              dV_pre_dx_t));
                        // End Slayer copy
                        for (size_t source = 0;
                             grad != 0.0 && source < TOTAL_NEURONS; source++) {
                            if (delays[source][dest] == 0.0) {
                                continue;
                            }

                            int delay       = delays[source][dest];
                            int source_time = t - delay;
                            if (source_time < 0) {
                                continue;
                            }

                            double source_spike = spikes[source][source_time];
                            delta_W[source][dest] += source_spike * grad;
                            spike_grad_history[source][source_time] +=
                                grad * weights[source][dest];
                            // Probably don't need to do this
                            // voltage_grad_history[source][source_time] +=
                            //     grad * weights[source][dest];
                        }

                        next_future_mem_grad[dest] =
                            (dL_dV * dV_post_dV_pre * dV_pre_dV_leak *
                             dV_leak_dV_t1) +
                            (dL_dV * dV_post_ds_t * ds_t_dV_pre *
                             dV_pre_dV_leak * dV_leak_dV_t1) +
                            (spike_grad_history[dest][t] * ds_t_dV_pre *
                             dV_pre_dV_leak * dV_leak_dV_t1);
                    }

                    memcpy(future_mem_grad, next_future_mem_grad,
                           sizeof(future_mem_grad));
                }
            }

            // Average gradients over the actual batch size.
            size_t current_batch_size =
                min((size_t)BATCH_SIZE, train.rows - (size_t)batch_start);
            double inv_batch = 1.0 / ((double)current_batch_size * TIMESTEPS);
            for (size_t i = 0; i < TOTAL_NEURONS; i++) {
                for (size_t j = 0; j < TOTAL_NEURONS; j++) {
                    delta_W[i][j] *= inv_batch;
                }
            }

            epoch_loss += batch_loss;
            correct += batch_correct;

            // Adam update
            b1_t *= BETA1;
            b2_t *= BETA2;

            for (size_t i = 0; i < TOTAL_NEURONS; i++) {
                for (size_t j = 0; j < n->get_node(i)->outgoing.size(); j++) {
                    Edge* e = n->get_node(i)->outgoing[j];
                    int k   = e->to->id;

                    m_weights[i][k] =
                        BETA1 * m_weights[i][k] + (1.0 - BETA1) * delta_W[i][k];
                    v_weights[i][k] =
                        BETA2 * v_weights[i][k] +
                        (1.0 - BETA2) * (delta_W[i][k] * delta_W[i][k]);

                    double mW_hat = m_weights[i][k] / (1.0 - b1_t);
                    double vW_hat = v_weights[i][k] / (1.0 - b2_t);

                    double lr = learning_rate;
                    if (epoch == 0) {
                        lr = ((batch_start + BATCH_SIZE) / (double)train.rows) *
                             learning_rate;
                    }

                    weights[i][k] -= lr * mW_hat / (sqrt(vW_hat + ADAM_EPS));
                    weights[i][k] -= lr * decay_rate * weights[i][k];
                    e->set("Weight", weights[i][k]);
                }
            }

            delete p;
        }

        // Training Metrics
        if (epoch_loss / (double)train.rows < best_train_loss) {
            best_train_loss = epoch_loss / (double)train.rows;
        }
        if (correct / (double)train.rows > best_train_acc) {
            best_train_acc = correct / (double)train.rows;
        }

        // Test
        double test_correct = 0.0;
        double test_loss    = 0.0;

        Processor* p = nullptr;
        load_network(&p, n);
        for (int i = 0; i < test.rows; i++) {
            p->clear_activity();

            for (size_t input = 0; input < INPUT_NEURONS; input++) {
                for (size_t t = 0; t < TIMESTEPS; t++) {
                    p->apply_spike({(int)input, (double)t,
                                    (double)test_spikes[i][input][t]});
                }
            }

            p->run(TIMESTEPS);
            const vector<int>& output_counts = p->output_counts();
            double logits[OUTPUT_NEURONS]    = {0.0};
            size_t max_idx                   = 0;
            int max_val                      = 0;
            for (size_t output = 0; output < OUTPUT_NEURONS; output++) {
                logits[output] = output_counts[output] / (double)TIMESTEPS;

                if (output_counts[output] > max_val) {
                    max_val = output_counts[output];
                    max_idx = output;
                }
            }
            double softmax_out[OUTPUT_NEURONS] = {0.0};
            softmax(logits, softmax_out, 3);

            test_loss -= log(softmax_out[(size_t)test.labels[i]]);
            if (max_idx == (size_t)test.labels[i]) {
                test_correct++;
            }
        }
        delete p;

        test_correct /= test.rows;
        test_loss /= test.rows;

        if (test_correct > best_test_acc) {
            best_test_acc = test_correct;
        }
        if (test_loss < best_test_loss) {
            best_test_loss = test_loss;
        }

        printf(
            "Epoch: %zu/%zu, Loss: %10g (Best: %10g), Acc: %10g (Best: "
            "%10g), TestLoss: %10g (Best: %10g), TestAcc: %10g (Best: %10g)\n ",
            epoch + 1, (size_t)EPOCHS, epoch_loss / (double)train.rows,
            best_train_loss, correct / (double)train.rows, best_train_acc,
            test_loss, best_test_loss, test_correct, best_test_acc);
        // printf("%g\n", best_loss);
    }

    // puts("Weights:");
    // for (size_t i = 0; i < TOTAL_NEURONS; i++) {
    //     for (size_t j = 0; j < TOTAL_NEURONS; j++) {
    //         printf("%12g ", weights[i][j]);
    //     }
    //     puts("");
    // }
    // puts("");

    delete n;
    free(train.data);
    free(train.labels);
    free(train.min_vals);
    free(train.max_vals);
    free(batch_order);
}
