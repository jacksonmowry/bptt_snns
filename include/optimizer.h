#pragma once

#include "csv.h"
#include "shared.h"
#include <cstddef>
#include <vector>

void weight_updates(const NetworkConfiguration* nc, const Dataset* d,
                    size_t current_batch_size, size_t batch_size,
                    size_t batch_start, size_t epoch, double& b1_t,
                    double& b2_t, std::vector<std::vector<double>>& m_weights,
                    std::vector<std::vector<double>>& v_weights,
                    double learning_rate, double decay_rate,
                    std::vector<std::vector<double>>& weights,
                    std::vector<std::vector<double>>& delta_W);
