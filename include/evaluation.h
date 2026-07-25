#pragma once

#include "cli.h"
#include "csv.h"
#include "framework.hpp"
#include <cstddef>


int evaluate_sample(neuro::Processor* p, const ClassificationDataset& dataset, size_t idx,
                    size_t hidden_neurons, size_t output_neurons,
                    size_t timesteps, bool timeseries, size_t input_neurons);

// Generate confusion matrix for the best saved network.
bool run_confusion_matrix(const CliConfig& cfg, const ClassificationDataset& train,
                          const ClassificationDataset& test, size_t input_neurons,
                          size_t hidden_neurons, size_t output_neurons,
                          size_t timesteps, bool timeseries);
