#ifndef FORWARD_PASS_H
#define FORWARD_PASS_H

#include "bptt_types.h"
#include "framework.hpp"
#include <cstddef>

EvaluationResults forward(TrainingBundle* tb, neuro::Processor* p,
                          const Dataset* d, size_t index,
                          const NetworkConfiguration* nc);

#endif // FORWARD_PASS_H
