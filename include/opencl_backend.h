#pragma once

#include "backend.h"
#include "opencl.hpp"
#include <chrono>
#include <memory>
#include <utility>
#include <vector>

class OpenclBackend : public TrainingBackend {
  public:
    OpenclBackend(const CliConfig& cfg, NetworkConfiguration& nc,
                  const Dataset& train, const Dataset& test,
                  size_t max_incoming, size_t max_outgoing);
    ~OpenclBackend() override;

    void do_one_epoch(size_t epoch) override;
    TrainingStats get_stats() const override;
    void update_weights(neuro::Network* network) override;

  private:
    const CliConfig& cfg;
    NetworkConfiguration& nc;
    const Dataset& train;
    const Dataset& test;
    size_t max_incoming;
    size_t max_outgoing;
    size_t batch_size;
    double learning_rate;
    double decay_rate;
    double rho;
    double tau;

    // GPU buffers
    std::unique_ptr<Memory<short>> x;
    std::unique_ptr<Memory<double>> data;
    std::unique_ptr<Memory<double>> test_data;
    std::unique_ptr<Memory<short>> v_thresh;
    std::unique_ptr<Memory<short>> weights;
    std::unique_ptr<Memory<uint>> delays;
    std::unique_ptr<Memory<uint>> incoming;
    std::unique_ptr<Memory<uint>> incoming_ids;
    std::unique_ptr<Memory<uchar>> is_input_neuron;
    std::unique_ptr<Memory<uchar>> is_output_neuron;
    std::unique_ptr<Memory<int>> v;
    std::unique_ptr<Memory<char>> s;
    std::unique_ptr<Memory<int>> v_pre;
    std::unique_ptr<Memory<float>> dL_ds;
    std::unique_ptr<Memory<float>> correct;
    std::unique_ptr<Memory<float>> loss;
    std::unique_ptr<Memory<float>> spike_grad_history;
    std::unique_ptr<Memory<float>> future_mem_grad;
    std::unique_ptr<Memory<float>> delta_W;
    std::unique_ptr<Memory<float>> neuron_grad;
    std::unique_ptr<Memory<float>> m_weights;
    std::unique_ptr<Memory<float>> v_weights;
    std::unique_ptr<Memory<uint>> outgoing;
    std::unique_ptr<Memory<uint>> gradient_slot;
    std::unique_ptr<Memory<float>> gradient_accumulators;

    // Kernels
    std::unique_ptr<Kernel> encode_kernel;
    std::unique_ptr<Kernel> encode_timeseries_kernel;
    std::unique_ptr<Kernel> forward_kernel;
    std::unique_ptr<Kernel> loss_kernel;
    std::unique_ptr<Kernel> backward_grad_kernel;
    std::unique_ptr<Kernel> backward_delta_w_kernel;
    std::unique_ptr<Kernel> weight_updates_kernel;

    // Batch ordering
    std::vector<size_t> batch_order;

    // Adam state
    double b1_t;
    double b2_t;

    TrainingStats stats = {0.0, 0.0, 0.0, 0.0};

    // Timing
    std::chrono::high_resolution_clock::time_point t_start;
};
