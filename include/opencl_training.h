#pragma once

#include "cli.h"
#include "shared.h"
#include "training.h"
#include <cstddef>
#include <string>

// Run full OpenCL training loop. Returns true on success.
// Reads final weights into state->weights on return.
bool opencl_train(const CliConfig& cfg, neuro::Network* n,
                  const NetworkConfiguration& nc, const Dataset& train,
                  const Dataset& test, TrainingState* state,
                  size_t max_incoming, size_t max_outgoing, size_t epochs,
                  size_t batch_size, double learning_rate, double decay_rate,
                  double rho, double tau, bool export_json);
