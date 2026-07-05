#ifndef BACKWARD_PASS_H
#define BACKWARD_PASS_H

#include <cstddef>
#include "bptt_types.h"

void backward(TrainingBundle* tb, const NetworkConfiguration* nc);

#endif // BACKWARD_PASS_H
