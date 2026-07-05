#ifndef BPTT_TYPES_H
#define BPTT_TYPES_H

#include "csv.h"
#include "framework.hpp"
#include <Eigen/Dense>
#include <vector>

// Adam/Learning Parameters
#define BETA1 (0.9)
#define BETA2 (0.999)
#define ADAM_EPS (1.0e-8)
#define NUM_LAYERS (3)

typedef struct {
    short v_decay;
    short v_rest;
    double tau_rho;
    double scale_rho;
    ushort num_neurons;
    ushort num_output_neurons;
    short num_steps;
    ushort max_incoming;
    double scale_factor;
} BwdParams;

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
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> vgh;
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
                   const std::vector<double>* thresholds)
        : weights(weights), delta_W(total_neurons), delays(delays),
          thresholds(thresholds),
          spikes(timesteps, std::vector<double>(total_neurons)),
          v_pre(timesteps, std::vector<double>(total_neurons)),
          spike_logits(output_neurons), target(output_neurons),
          dL_ds(output_neurons), softmax_out(output_neurons),
          future_mem_grad_(total_neurons), sgh(total_neurons, timesteps),
          vgh(total_neurons, timesteps), dL_dV_(total_neurons),
          v_pre_t_(total_neurons), dV_post_dV_pre_(total_neurons),
          dV_post_ds_t_(total_neurons), ds_t_dV_pre_(total_neurons),
          dV_leak_dV_t1_(total_neurons), grad_(total_neurons), rho(rho),
          tau(tau) {}
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

    size_t timesteps;
    bool timeseries;

    double min_potential;
    bool leak;
    double scale_factor;
    bool discrete;
};

struct ThreadArgs {
    TrainingBundle tb;
    NetworkConfiguration* nc;
    const size_t* order;
    const Dataset* train;
    const Dataset* test;
    int* max_idx;
    int* work_idx;
    int* done;
    double loss;
    size_t correct;
    size_t processed;
    pthread_mutex_t* mut;
    pthread_cond_t* have_work;
    pthread_cond_t* done_work;
    bool* train_p;
    bool* die;

    ThreadArgs(size_t total_neurons, size_t timesteps, size_t output_neurons,
               double rho, double tau,
               const std::vector<std::vector<double>>* weights,
               const std::vector<std::vector<int>>* delays,
               const std::vector<double>* thresholds, NetworkConfiguration* nc,
               const size_t* order, const Dataset* train, const Dataset* test,
               int* max_idx, int* work_idx, int* done, pthread_mutex_t* mut,
               pthread_cond_t* have_work, pthread_cond_t* done_work,
               bool* train_p, bool* die)
        : tb(total_neurons, timesteps, output_neurons, rho, tau, weights,
             delays, thresholds),
          nc(nc), order(order), train(train), test(test), max_idx(max_idx),
          work_idx(work_idx), done(done), mut(mut), have_work(have_work),
          done_work(done_work), train_p(train_p), die(die) {}
};

#endif // BPTT_TYPES_H
