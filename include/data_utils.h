#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <cstddef>
#include "framework.hpp"
#include "csv.h"
using namespace neuro;

void encode_spikes(Processor* p, const Dataset* d, size_t index,
                   size_t timesteps, bool timeseries, size_t input_neurons);
size_t label_count(const Dataset* d);

#endif // DATA_UTILS_H
