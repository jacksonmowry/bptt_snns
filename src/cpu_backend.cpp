#include "cpu_backend.h"
#include "forward_backward.h"
#include "network_setup.h"
#include "network_utils.h"
#include "optimizer.h"
#include "training.h"
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

using namespace std;

CpuBackend::CpuBackend(const CliConfig& cfg, NetworkConfiguration& nc,
                       const Dataset& train, const Dataset& test)
    : cfg(cfg), nc(nc), train(train), test(test), batch_size(cfg.batch_size),
      learning_rate(cfg.learning_rate), decay_rate(cfg.decay_rate) {
    size_t threads = cfg.threads;
    state          = init_training(nc, train, cfg.threads, cfg.rho, cfg.tau);
    init_network_weights(nc.n, nc.total_neurons, nc.discrete, nc.scale_factor,
                         state->weights, state->delays, state->thresholds);

    // Patch nc and dataset refs into ThreadArgs
    for (size_t i = 0; i < threads; i++) {
        state->tas[i].nc    = &nc;
        state->tas[i].train = &train;
        state->tas[i].test  = &test;
        pthread_create(state->tids + i, NULL, worker, (void*)(state->tas + i));
    }
}

CpuBackend::~CpuBackend() {
    pthread_mutex_lock(&state->mut);
    state->die = true;
    pthread_mutex_unlock(&state->mut);
    pthread_cond_broadcast(&state->have_work);

    for (size_t i = 0; i < cfg.threads; i++) {
        pthread_join(state->tids[i], NULL);
    }

    free(state->batch_order);
    free(state->tas);
    free(state->tids);
    delete state;
}

void CpuBackend::do_one_epoch(size_t epoch) {
    (void)epoch; // used by weight_updates() for LR scheduling
    size_t threads       = cfg.threads;
    size_t total_neurons = nc.total_neurons;

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
        size_t j              = rand() % train.observations;
        size_t tmp            = state->batch_order[i];
        state->batch_order[i] = state->batch_order[j];
        state->batch_order[j] = tmp;
    }
    pthread_mutex_unlock(&state->mut);

    // Batch processing loop
    for (int batch_start = 0; batch_start < train.observations;
         batch_start += batch_size) {
        size_t current_batch_size =
            min(batch_size, train.observations - (size_t)batch_start);

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

        weight_updates(&nc, &train, current_batch_size, batch_size, batch_start,
                       epoch, state->b1_t, state->b2_t, state->m_weights,
                       state->v_weights, learning_rate, decay_rate,
                       state->weights, state->delta_W);
    }

    // Training metrics
    double avg_train_loss = epoch_loss / (double)train.observations;
    double avg_train_acc  = correct / (double)train.observations;

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
    }

    stats.train_acc  = avg_train_acc;
    stats.train_loss = avg_train_loss;
    stats.test_acc   = (test.observations > 0) ? test_correct : 0.0;
    stats.test_loss  = (test.observations > 0) ? test_loss : 0.0;
}

TrainingStats CpuBackend::get_stats() const { return stats; }

void CpuBackend::update_weights(neuro::Network* network) {
    // CPU backend: weights already synced in edges by weight_updates()
    // Nothing extra needed.
    (void)network;
}
