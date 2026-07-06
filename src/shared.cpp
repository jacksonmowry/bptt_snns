#include "shared.h"
#include <Eigen/Dense>

TrainingBundle::TrainingBundle(size_t total_neurons, size_t timesteps,
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
