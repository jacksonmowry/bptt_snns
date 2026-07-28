#pragma once

#include "csv.h"
#include "framework.hpp"
#include <vector>

void encode_spikes(neuro::Processor* p, const ClassificationDataset* d, size_t index,
                   size_t timesteps, bool timeseries, size_t input_neurons);

/* Encode a single non-timeseries sample into a spike raster.
 * values: array of n_features values (already normalized to [0,1] OR raw with min/max).
 * If use_normalized is true, values are assumed [0,1] (no min/max lookup).
 * Returns [num_neurons][timesteps] boolean matrix where num_neurons = n_features * 2.
 */
std::vector<std::vector<bool>> encode_spike_raster(const double* values, size_t n_features,
                                                    size_t timesteps,
                                                    const double* min_vals,
                                                    const double* max_vals,
                                                    bool use_normalized);
