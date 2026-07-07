#include "training.h"
#include "cli.h"
#include "network_utils.h"
#include "forward_backward.h"
#include "optimizer.h"
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace std;
using nlohmann::json;

TrainingState* init_training(neuro::Network* n,
                             const NetworkConfiguration& nc,
                             const Dataset& train,
                             size_t threads,
                             double rho, double tau) {
    TrainingState* state = new TrainingState();
    size_t total_neurons = nc.total_neurons;

    // Init Adam buffers + delta_W
    state->m_weights.resize(total_neurons);
    state->v_weights.resize(total_neurons);
    state->delta_W.resize(total_neurons);

    for (size_t i = 0; i < total_neurons; i++) {
        size_t inc = n->get_node(i)->incoming.size();
        state->m_weights[i].resize(inc);
        state->v_weights[i].resize(inc);
        state->delta_W[i].resize(inc);
    }

    // Threading
    state->tas          = (ThreadArgs*)calloc(threads, sizeof(*state->tas));
    state->tids         = (pthread_t*)calloc(threads, sizeof(*state->tids));
    state->batch_order  = (size_t*)calloc(train.observations, sizeof(size_t));

    for (size_t i = 0; i < threads; i++) {
        state->tas[i] = ThreadArgs(
            total_neurons, nc.timesteps, nc.output_neurons, rho, tau,
            &state->weights, &state->delays, &state->thresholds,
            nullptr, // nc set later
            state->batch_order, nullptr, nullptr, // train/test set later
            &state->max_idx, &state->work_idx, &state->done_count,
            &state->mut, &state->have_work, &state->done_work,
            &state->train_p, &state->die);

        for (size_t neuron = 0; neuron < total_neurons; neuron++) {
            state->tas[i].tb.delta_W[neuron].resize(
                n->get_node(neuron)->incoming.size());
        }
    }

    // Patch nc and dataset refs into ThreadArgs
    for (size_t i = 0; i < threads; i++) {
        // Will be set from caller
    }

    return state;
}

void run_training(const CliConfig& cfg,
                  neuro::Network* n,
                  NetworkConfiguration& nc,
                  const Dataset& train,
                  const Dataset& test,
                  TrainingState* state,
                  size_t epochs, size_t batch_size,
                  double learning_rate, double decay_rate) {
    size_t threads = cfg.threads;
    size_t total_neurons = nc.total_neurons;

    // Patch nc and dataset refs into ThreadArgs
    for (size_t i = 0; i < threads; i++) {
        state->tas[i].nc = &nc;
        state->tas[i].train = &train;
        state->tas[i].test = &test;
        pthread_create(state->tids + i, NULL, worker, (void*)(state->tas + i));
    }

    puts("Beginning training");

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss = 0.0;
        size_t correct    = 0;

        // Reset work index before each epoch
        pthread_mutex_lock(&state->mut);
        state->work_idx = 0;
        state->train_p  = true;

        // Shuffle the batch order for randomness
        for (int i = 0; i < train.observations; i++) {
            state->batch_order[i] = i;
        }
        for (int i = 0; i < train.observations; i++) {
            size_t j       = rand() % train.observations;
            size_t tmp     = state->batch_order[i];
            state->batch_order[i] = state->batch_order[j];
            state->batch_order[j] = tmp;
        }
        pthread_mutex_unlock(&state->mut);

        // Batch processing loop
        for (int batch_start = 0; batch_start < train.observations;
             batch_start += batch_size) {
            size_t current_batch_size = min(
                (size_t)batch_size, train.observations - (size_t)batch_start);

            pthread_mutex_lock(&state->mut);
            state->work_idx   = batch_start;
            state->done_count = 0;
            state->max_idx    = batch_start + current_batch_size;
            pthread_cond_broadcast(&state->have_work);
            pthread_mutex_unlock(&state->mut);

            pthread_mutex_lock(&state->mut);
            while (state->done_count < (int)current_batch_size) {
                pthread_cond_wait(&state->done_work, &state->mut);
            }
            pthread_mutex_unlock(&state->mut);

            for (size_t i = 0; i < threads; i++) {
                epoch_loss += state->tas[i].loss;
                correct += state->tas[i].correct;
                state->tas[i].loss      = 0;
                state->tas[i].correct   = 0;
                state->tas[i].processed = 0;

                for (size_t row = 0; row < total_neurons; row++) {
                    for (size_t incoming = 0;
                         incoming < state->tas[i].tb.delta_W[row].size();
                         incoming++) {
                        state->delta_W[row][incoming] +=
                            state->tas[i].tb.delta_W[row][incoming];
                        state->tas[i].tb.delta_W[row][incoming] = 0.0;
                    }
                }
            }

            weight_updates(&nc, &train, current_batch_size, batch_size,
                           batch_start, epoch, state->b1_t, state->b2_t,
                           state->m_weights, state->v_weights,
                           learning_rate, decay_rate,
                           state->weights, state->delta_W);
        }

        // Training metrics
        double avg_train_loss = epoch_loss / (double)train.observations;
        double avg_train_acc  = correct / (double)train.observations;
        if (avg_train_loss < state->best_train_loss) {
            state->best_train_loss = avg_train_loss;
        }
        if (avg_train_acc > state->best_train_acc) {
            state->best_train_acc = avg_train_acc;
        }

        // Test
        double test_correct = 0.0;
        double test_loss    = 0.0;

        pthread_mutex_lock(&state->mut);
        state->work_idx   = 0;
        state->done_count = 0;
        state->max_idx    = test.observations;
        state->train_p    = false;
        pthread_cond_broadcast(&state->have_work);
        pthread_mutex_unlock(&state->mut);

        pthread_mutex_lock(&state->mut);
        while (state->done_count < test.observations) {
            pthread_cond_wait(&state->done_work, &state->mut);
        }
        pthread_mutex_unlock(&state->mut);

        pthread_mutex_lock(&state->mut);
        for (size_t i = 0; i < threads; i++) {
            test_loss += state->tas[i].loss;
            test_correct += state->tas[i].correct;
            state->tas[i].loss      = 0;
            state->tas[i].correct   = 0;
            state->tas[i].processed = 0;
        }
        state->max_idx = -1;
        pthread_mutex_unlock(&state->mut);

        if (test.observations > 0) {
            test_correct /= test.observations;
            test_loss /= test.observations;

            if (test_correct > state->best_test_acc) {
                state->best_test_acc = test_correct;
            }
            if (test_loss < state->best_test_loss) {
                state->best_test_loss = test_loss;
            }
        }

        printf(
            "Epoch: %4zu/%zu, Loss: %10g (Best: %10g), Acc: %10g (Best: %10g), "
            "TestLoss: %10g (Best: %10g), TestAcc: %10g (Best: %10g)\n",
            epoch + 1, epochs, avg_train_loss, state->best_train_loss,
            avg_train_acc, state->best_train_acc,
            test_loss, state->best_test_loss,
            test_correct, state->best_test_acc);

        if (!cfg.network_json_out.empty()) {
            json meta               = n->get_data("other");
            meta["best_train_loss"] = state->best_train_loss;
            meta["best_test_loss"]  = state->best_test_loss;
            meta["best_train_acc"]  = state->best_train_acc;
            meta["best_test_acc"]   = state->best_test_acc;
            meta["epoch"]           = epoch + 1;
            n->set_data("other", meta);

            json j;
            n->to_json(j);
            std::ofstream fout(cfg.network_json_out);
            if (!fout) {
                fprintf(stderr,
                        "Failed to open networks/trained.json for writing\n");
                exit(1);
            }
            fout << j << std::endl;
            fout.close();
        }
    }
}

void cleanup_training(TrainingState* state, size_t threads) {
    pthread_mutex_lock(&state->mut);
    state->die = true;
    pthread_mutex_unlock(&state->mut);
    pthread_cond_broadcast(&state->have_work);

    for (size_t i = 0; i < threads; i++) {
        pthread_join(state->tids[i], NULL);
    }

    free(state->batch_order);
    free(state->tas);
    free(state->tids);
    delete state;
}
