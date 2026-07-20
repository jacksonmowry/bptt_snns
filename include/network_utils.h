#pragma once

#include "cli.h"
#include "framework.hpp"

void load_network(neuro::Processor** pp, neuro::Network* net);
void export_network(neuro::Network* n, const CliConfig& cfg,
                    double best_train_acc, double best_train_loss,
                    double best_test_acc, double best_test_loss,
                    const char** label_strings, int label_strings_count);
