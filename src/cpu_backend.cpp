#include "cpu_backend.h"
#include "forward_backward.h"
#include "optimizer.h"
#include "network_utils.h"
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace std;
using nlohmann::json;

CpuBackend::CpuBackend(const CliConfig& cfg, neuro::Network* n,
                       NetworkConfiguration& nc, const Dataset& train,
                       const Dataset& test, TrainingState* state,
                       size_t batch_size, double learning_rate,
                       double decay_rate, bool export_json)
    : m_cfg(cfg), m_n(n), m_nc(nc), m_train(train), m_test(test),
      m_state(state), m_batch_size(batch_size), m_learning_rate(learning_rate),
      m_decay_rate(decay_rate), m_export_json(export_json) {
    size_t threads = cfg.threads;
    size_t total_neurons = nc.total_neurons;

    // Patch nc and dataset refs into ThreadArgs
    for (size_t i = 0; i < threads; i++) {
        state->tas[i].nc = &nc;
        state->tas[i].train = &train;
        state->tas[i].test = &test;
        pthread_create(state->tids + i, NULL, worker, (void*)(state->tas + i));
    }
}

CpuBackend::~CpuBackend() {
    pthread_mutex_lock(&m_state->mut);
    m_state->die = true;
    pthread_mutex_unlock(&m_state->mut);
    pthread_cond_broadcast(&m_state->have_work);

    for (size_t i = 0; i < m_cfg.threads; i++) {
        pthread_join(m_state->tids[i], NULL);
    }
}

void CpuBackend::do_one_epoch(size_t epoch) {
    size_t threads = m_cfg.threads;
    size_t total_neurons = m_nc.total_neurons;

    double epoch_loss = 0.0;
    size_t correct = 0;

    // Reset work index before each epoch
    pthread_mutex_lock(&m_state->mut);
    m_state->work_idx = 0;
    m_state->train_p = true;

    // Shuffle the batch order for randomness
    for (int i = 0; i < m_train.observations; i++) {
        m_state->batch_order[i] = i;
    }
    for (int i = 0; i < m_train.observations; i++) {
        size_t j = rand() % m_train.observations;
        size_t tmp = m_state->batch_order[i];
        m_state->batch_order[i] = m_state->batch_order[j];
        m_state->batch_order[j] = tmp;
    }
    pthread_mutex_unlock(&m_state->mut);

    // Batch processing loop
    for (int batch_start = 0; batch_start < m_train.observations;
         batch_start += m_batch_size) {
        size_t current_batch_size = min(
            m_batch_size, m_train.observations - (size_t)batch_start);

        pthread_mutex_lock(&m_state->mut);
        m_state->work_idx = batch_start;
        m_state->done_count = 0;
        m_state->max_idx = batch_start + current_batch_size;
        pthread_cond_broadcast(&m_state->have_work);
        pthread_mutex_unlock(&m_state->mut);

        pthread_mutex_lock(&m_state->mut);
        while (m_state->done_count < (int)current_batch_size) {
            pthread_cond_wait(&m_state->done_work, &m_state->mut);
        }
        pthread_mutex_unlock(&m_state->mut);

        for (size_t i = 0; i < threads; i++) {
            epoch_loss += m_state->tas[i].loss;
            correct += m_state->tas[i].correct;
            m_state->tas[i].loss = 0;
            m_state->tas[i].correct = 0;
            m_state->tas[i].processed = 0;

            for (size_t row = 0; row < total_neurons; row++) {
                for (size_t incoming = 0;
                     incoming < m_state->tas[i].tb.delta_W[row].size();
                     incoming++) {
                    m_state->delta_W[row][incoming] +=
                        m_state->tas[i].tb.delta_W[row][incoming];
                    m_state->tas[i].tb.delta_W[row][incoming] = 0.0;
                }
            }
        }

        weight_updates(&m_nc, &m_train, current_batch_size, m_batch_size,
                       batch_start, epoch, m_state->b1_t, m_state->b2_t,
                       m_state->m_weights, m_state->v_weights,
                       m_learning_rate, m_decay_rate,
                       m_state->weights, m_state->delta_W);
    }

    // Training metrics
    double avg_train_loss = epoch_loss / (double)m_train.observations;
    double avg_train_acc = correct / (double)m_train.observations;
    if (avg_train_loss < m_state->best_train_loss) {
        m_state->best_train_loss = avg_train_loss;
    }
    if (avg_train_acc > m_state->best_train_acc) {
        m_state->best_train_acc = avg_train_acc;
    }

    // Test
    double test_correct = 0.0;
    double test_loss = 0.0;

    pthread_mutex_lock(&m_state->mut);
    m_state->work_idx = 0;
    m_state->done_count = 0;
    m_state->max_idx = m_test.observations;
    m_state->train_p = false;
    pthread_cond_broadcast(&m_state->have_work);
    pthread_mutex_unlock(&m_state->mut);

    pthread_mutex_lock(&m_state->mut);
    while (m_state->done_count < m_test.observations) {
        pthread_cond_wait(&m_state->done_work, &m_state->mut);
    }
    pthread_mutex_unlock(&m_state->mut);

    pthread_mutex_lock(&m_state->mut);
    for (size_t i = 0; i < threads; i++) {
        test_loss += m_state->tas[i].loss;
        test_correct += m_state->tas[i].correct;
        m_state->tas[i].loss = 0;
        m_state->tas[i].correct = 0;
        m_state->tas[i].processed = 0;
    }
    m_state->max_idx = -1;
    pthread_mutex_unlock(&m_state->mut);

    if (m_test.observations > 0) {
        test_correct /= m_test.observations;
        test_loss /= m_test.observations;

        if (test_correct > m_state->best_test_acc) {
            m_state->best_test_acc = test_correct;
        }
        if (test_loss < m_state->best_test_loss) {
            m_state->best_test_loss = test_loss;
        }
    }

    m_stats.train_acc = avg_train_acc;
    m_stats.train_loss = avg_train_loss;
    m_stats.test_acc = (m_test.observations > 0) ? test_correct : 0.0;
    m_stats.test_loss = (m_test.observations > 0) ? test_loss : 0.0;

    printf(
        "Epoch: %4zu/%zu, Loss: %10g (Best: %10g), Acc: %10g (Best: %10g), "
        "TestLoss: %10g (Best: %10g), TestAcc: %10g (Best: %10g)\n",
        epoch + 1, m_cfg.epochs, avg_train_loss, m_state->best_train_loss,
        avg_train_acc, m_state->best_train_acc,
        test_loss, m_state->best_test_loss,
        test_correct, m_state->best_test_acc);
}

void CpuBackend::finalize() {
    if (m_export_json && !m_cfg.network_json_out.empty()) {
        // Inject final metadata before export
        json meta = m_n->get_data("other");
        meta["best_train_loss"] = m_state->best_train_loss;
        meta["best_test_loss"] = m_state->best_test_loss;
        meta["best_train_acc"] = m_state->best_train_acc;
        meta["best_test_acc"] = m_state->best_test_acc;
        meta["epoch"] = m_cfg.epochs;
        m_n->set_data("other", meta);

        json j;
        m_n->to_json(j);
        std::ofstream fout(m_cfg.network_json_out);
        if (!fout) {
            fprintf(stderr, "Failed to open %s for writing\n",
                    m_cfg.network_json_out.c_str());
            exit(1);
        }
        fout << j.dump(2) << std::endl;
        fout.close();
    }
}

TrainingStats CpuBackend::get_stats() const { return m_stats; }

void CpuBackend::update_weights(neuro::Network* network) {
    // CPU backend: weights already synced in edges by weight_updates()
    // Nothing extra needed.
    (void)network;
}
