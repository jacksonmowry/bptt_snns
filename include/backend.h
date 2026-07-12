#pragma once

#include "shared.h"
#include "training.h"
#include <cstddef>
#include <memory>
#include <utility>

struct TrainingStats {
    double train_acc;
    double train_loss;
    double test_acc;
    double test_loss;
};

class TrainingBackend {
  public:
    virtual ~TrainingBackend() = default;

    virtual void do_one_epoch(size_t epoch)              = 0;
    virtual TrainingStats get_stats() const              = 0;
    virtual void update_weights(neuro::Network* network) = 0;
};

std::unique_ptr<TrainingBackend>
create_backend(const CliConfig& cfg, neuro::Network* n,
               NetworkConfiguration& nc, const Dataset& train,
               const Dataset& test, TrainingState* state, size_t batch_size,
               double learning_rate, double decay_rate, double rho, double tau);
