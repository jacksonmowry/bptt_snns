#pragma once

#include "cli.h"
#include "csv.h"
#include "framework.hpp"
#include "shared.h"
#include <cstddef>
#include <string>
#include <vector>

// Populate parameters from saved metadata (or leave defaults for new network)
// Returns pointer to owned Network.
neuro::Network*
load_and_init_network(const std::string& json_file, double& connectivity,
                      double& learning_rate, double& decay_rate, double& tau,
                      double& rho, size_t& timesteps, size_t& hidden_neurons,
                      unsigned long& seed, size_t& epochs, size_t& batch_size,
                      double& training_percent, size_t& threads,
                      bool& timeseries);

// Build and attach full reproducibility metadata to the network.
void build_run_metadata(neuro::Network* n, int argc, char* argv[],
                        const CliConfig& cfg, size_t input_neurons,
                        size_t output_neurons, size_t total_neurons,
                        size_t neuron_count, size_t synapse_count,
                        bool discrete, double min_potential, double min_weight,
                        double max_weight, double max_threshold,
                        const std::string& leak_prop, int scale,
                        double scale_factor);

// Generate nodes/edges for an empty network. Returns (neuron_count,
// synapse_count).
std::pair<size_t, size_t>
generate_network(neuro::Network* n, size_t input_neurons, size_t hidden_neurons,
                 size_t output_neurons, size_t total_neurons,
                 double connectivity, bool discrete, int scale,
                 double scale_factor, double min_weight, double max_weight,
                 double max_threshold);

// Extract weights, delays, thresholds from Network into flat-accessible arrays.
void init_network_weights(neuro::Network* n, size_t total_neurons,
                          bool discrete, double scale_factor,
                          std::vector<std::vector<double>>& weights,
                          std::vector<std::vector<int>>& delays,
                          std::vector<double>& thresholds);
