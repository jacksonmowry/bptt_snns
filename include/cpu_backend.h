#pragma once

#include "backend.h"
#include "threading.h"
#include <memory>
#include <utility>

class CpuBackend : public TrainingBackend {
public:
    CpuBackend(const CliConfig& cfg, neuro::Network* n,
               NetworkConfiguration& nc, const Dataset& train,
               const Dataset& test, TrainingState* state,
               size_t batch_size, double learning_rate, double decay_rate);
    ~CpuBackend() override;

    void do_one_epoch(size_t epoch) override;
    TrainingStats get_stats() const override;
    void update_weights(neuro::Network* network) override;

private:
    const CliConfig& m_cfg;
    NetworkConfiguration& m_nc;
    const Dataset& m_train;
    const Dataset& m_test;
    TrainingState* m_state;
    size_t m_batch_size;
    double m_learning_rate;
    double m_decay_rate;
    TrainingStats m_stats = {0.0, 0.0, 0.0, 0.0, 0.0, 1e18, 0.0, 1e18};
};
