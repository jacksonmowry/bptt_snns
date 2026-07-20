#pragma once

#include "backend.h"
#include "threading.h"
#include <memory>
#include <utility>

class CpuBackend : public TrainingBackend {
  public:
    CpuBackend(const CliConfig& cfg, NetworkConfiguration& nc,
               const Dataset& train, const Dataset& test);
    ~CpuBackend() override;

    void do_one_epoch(size_t epoch) override;
    TrainingStats get_stats() const override;
    void update_weights(neuro::Network* network) override;

  private:
    const CliConfig& cfg;
    NetworkConfiguration& nc;
    const Dataset& train;
    const Dataset& test;
    TrainingState* state;
    size_t batch_size;
    double learning_rate;
    double decay_rate;
    TrainingStats stats = {0.0, 0.0, 0.0, 0.0};
};
