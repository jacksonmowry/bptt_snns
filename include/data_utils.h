#pragma once

#include "csv.h"
#include "framework.hpp"

size_t label_count(const ClassificationDataset* d);
void encode_spikes(neuro::Processor* p, const ClassificationDataset* d, size_t index,
                   size_t timesteps, bool timeseries, size_t input_neurons);
