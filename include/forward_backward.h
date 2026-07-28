#pragma once

#include "csv.h"
#include "framework.hpp"
#include "shared.h"
#include <cstddef>

EvaluationResults forward(TrainingBundle* tb, neuro::Processor* p,
                          const Dataset* d, size_t index,
                          const NetworkConfiguration* nc);
void backward(TrainingBundle* tb, const NetworkConfiguration* nc);
