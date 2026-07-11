#pragma once

#include "backend.h"
#include "opencl.hpp"
#include <chrono>
#include <memory>
#include <utility>
#include <vector>

class OpenclBackend : public TrainingBackend {
public:
    OpenclBackend(const CliConfig& cfg, neuro::Network* n,
                  NetworkConfiguration& nc, const Dataset& train,
                  const Dataset& test, TrainingState* state,
                  size_t max_incoming, size_t max_outgoing,
                  size_t batch_size, double learning_rate,
                  double decay_rate, double rho, double tau);
    ~OpenclBackend() override;

    void do_one_epoch(size_t epoch) override;
    TrainingStats get_stats() const override;
    void update_weights(neuro::Network* network) override;

    // Run CPU test eval using GPU weights
    std::pair<double, double> run_final_cpu_eval();

private:
    const CliConfig& m_cfg;
    neuro::Network* m_n;
    NetworkConfiguration& m_nc;
    const Dataset& m_train;
    const Dataset& m_test;
    TrainingState* m_state;
    size_t m_max_incoming;
    size_t m_max_outgoing;
    size_t m_batch_size;
    double m_learning_rate;
    double m_decay_rate;
    double m_rho;
    double m_tau;


    // GPU buffers
    std::unique_ptr<Memory<short>> m_x;
    std::unique_ptr<Memory<double>> m_data;
    std::unique_ptr<Memory<double>> m_test_data;
    std::unique_ptr<Memory<short>> m_v_thresh;
    std::unique_ptr<Memory<short>> m_weights;
    std::unique_ptr<Memory<uint>> m_delays;
    std::unique_ptr<Memory<uint>> m_incoming;
    std::unique_ptr<Memory<uint>> m_incoming_ids;
    std::unique_ptr<Memory<uchar>> m_is_input_neuron;
    std::unique_ptr<Memory<uchar>> m_is_output_neuron;
    std::unique_ptr<Memory<short>> m_v;
    std::unique_ptr<Memory<char>> m_s;
    std::unique_ptr<Memory<short>> m_v_pre;
    std::unique_ptr<Memory<float>> m_dL_ds;
    std::unique_ptr<Memory<float>> m_correct;
    std::unique_ptr<Memory<float>> m_loss;
    std::unique_ptr<Memory<float>> m_spike_grad_history;
    std::unique_ptr<Memory<float>> m_future_mem_grad;
    std::unique_ptr<Memory<float>> m_delta_W;
    std::unique_ptr<Memory<float>> m_neuron_grad;
    std::unique_ptr<Memory<float>> m_m_weights;
    std::unique_ptr<Memory<float>> m_v_weights;
    std::unique_ptr<Memory<uint>> m_outgoing;
    std::unique_ptr<Memory<uint>> m_gradient_slot;
    std::unique_ptr<Memory<float>> m_gradient_accumulators;

    // Kernels
    std::unique_ptr<Kernel> m_encode_kernel;
    std::unique_ptr<Kernel> m_forward_kernel;
    std::unique_ptr<Kernel> m_loss_kernel;
    std::unique_ptr<Kernel> m_backward_grad_kernel;
    std::unique_ptr<Kernel> m_backward_delta_w_kernel;
    std::unique_ptr<Kernel> m_weight_updates_kernel;

    // Batch ordering
    std::vector<size_t> m_batch_order;

    // Adam state
    double m_b1_t;
    double m_b2_t;

    TrainingStats m_stats = {0.0, 0.0, 0.0, 0.0, 0.0, 1e18, 0.0, 1e18};

    // Timing
    std::chrono::high_resolution_clock::time_point m_t_start;
};
