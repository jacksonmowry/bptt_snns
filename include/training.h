#pragma once

#include "cli.h"
#include "shared.h"
#include "threading.h"
#include <cstddef>
#include <string>
#include <vector>

// Holds all mutable training state
struct TrainingState {
    // Weights
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<int>> delays;
    std::vector<double> thresholds;
    // Adam buffers
    std::vector<std::vector<double>> m_weights;
    std::vector<std::vector<double>> v_weights;
    std::vector<std::vector<double>> delta_W;
    double b1_t = 1.0;
    double b2_t = 1.0;
    // Threading
    ThreadArgs* tas          = nullptr;
    pthread_t* tids          = nullptr;
    int max_idx              = -1;
    pthread_mutex_t mut      = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t have_work = PTHREAD_COND_INITIALIZER;
    pthread_cond_t done_work = PTHREAD_COND_INITIALIZER;
    bool train_p             = true;
    bool die                 = false;
    size_t* batch_order      = nullptr;
    int work_idx             = 0;
    int done_count           = 0;
    // Best metrics
    double best_train_loss = 1e18;
    double best_test_loss  = 1e18;
    double best_train_acc  = 0.0;
    double best_test_acc   = 0.0;
};

// Allocate and initialize training state + threads.
TrainingState* init_training(neuro::Network* n, const NetworkConfiguration& nc,
                             const Dataset& train, size_t threads, double rho,
                             double tau);

// Run full training loop (epochs, batches, metrics, save).
void run_training(const CliConfig& cfg, neuro::Network* n,
                  NetworkConfiguration& nc, const Dataset& train,
                  const Dataset& test, TrainingState* state, size_t epochs,
                  size_t batch_size, double learning_rate, double decay_rate);

// Stop threads and free all training resources.
void cleanup_training(TrainingState* state, size_t threads);
