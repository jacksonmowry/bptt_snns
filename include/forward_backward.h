#pragma once

#include <cstddef>
#include "csv.h"
#include "shared.h"
#include "framework.hpp"

EvaluationResults forward(TrainingBundle* tb, neuro::Processor* p, const Dataset* d,
                          size_t index, const NetworkConfiguration* nc);
void backward(TrainingBundle* tb, const NetworkConfiguration* nc);
