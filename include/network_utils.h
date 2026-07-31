#pragma once

#include "cli.h"
#include "framework.hpp"
#include <string>
#include <vector>

void load_network(neuro::Processor** pp, neuro::Network* net);
void export_network(neuro::Network* n, const CliConfig& cfg,
                    double best_train_acc, double best_train_loss,
                    double best_test_acc, double best_test_loss,
                    const std::vector<std::string>& label_strings,
                    bool is_regression, size_t current_epoch);
