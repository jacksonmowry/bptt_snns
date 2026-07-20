#pragma once

#include "framework.hpp"
#include <Eigen/Dense>
#include <cstddef>
#include <vector>

struct TrainingBundle {
    const std::vector<std::vector<double>>* weights;
    std::vector<std::vector<double>> delta_W;
    const std::vector<std::vector<int>>* delays;
    const std::vector<double>* thresholds;

    std::vector<std::vector<double>> spikes;
    std::vector<std::vector<double>> v_pre;
    std::vector<double> spike_logits;
    std::vector<double> target;
    std::vector<double> dL_ds;
    std::vector<double> softmax_out;

    Eigen::VectorXd future_mem_grad_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> sgh;
    Eigen::VectorXd dL_dV_;
    Eigen::VectorXd v_pre_t_;
    Eigen::VectorXd dV_post_dV_pre_;
    Eigen::VectorXd dV_post_ds_t_;
    Eigen::VectorXd ds_t_dV_pre_;
    Eigen::VectorXd dV_leak_dV_t1_;
    Eigen::VectorXd grad_;

    double rho;
    double tau;

    TrainingBundle(size_t total_neurons, size_t timesteps,
                   size_t output_neurons, double rho, double tau,
                   const std::vector<std::vector<double>>* weights,
                   const std::vector<std::vector<int>>* delays,
                   const std::vector<double>* thresholds);
};

struct EvaluationResults {
    double correct;
    double loss;
};

struct NetworkConfiguration {
    neuro::Network* n;

    size_t input_neurons;
    size_t hidden_neurons;
    size_t output_neurons;
    size_t layer_offsets[3];
    size_t total_neurons;
    size_t max_incoming;
    size_t max_outgoing;

    size_t timesteps;
    bool timeseries;

    double min_potential;
    bool leak;
    double scale_factor;
    int steps;
    bool discrete;
    double min_weight;
    double max_weight;
    double spike_value_factor;
};
