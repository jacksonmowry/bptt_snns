#include "evaluation.h"
#include "data_utils.h"
#include "framework.hpp"
#include "network_utils.h"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
using namespace std;
using namespace neuro;

/* Run inference on a single sample and return predicted class index */
int evaluate_sample(Processor* p, const Dataset& dataset, size_t idx,
                    size_t hidden_neurons, size_t output_neurons,
                    size_t timesteps, bool timeseries, size_t input_neurons) {
    p->clear_activity();

    /* Reuse encode_spikes from data_utils */
    encode_spikes(p, &dataset, idx, timesteps, timeseries, input_neurons);

    /* Accumulate output counts across all timesteps */
    vector<int> output_logits(output_neurons, 0);
    for (size_t t = 0; t < timesteps; t++) {
        p->run(1);
        const vector<int>& neuron_counts = p->neuron_counts();
        for (int o = 0; o < (int)output_neurons; o++) {
            size_t output_neuron_idx = input_neurons + hidden_neurons + o;
            output_logits[o] += neuron_counts[output_neuron_idx];
        }
    }

    /* Find argmax */
    int pred      = 0;
    int max_count = output_logits[0];
    for (int o = 1; o < (int)output_neurons; o++) {
        if (output_logits[o] > max_count) {
            max_count = output_logits[o];
            pred      = o;
        }
    }
    return pred;
}

bool run_confusion_matrix(const CliConfig& cfg, const Dataset& train,
                          const Dataset& test,
                          const std::vector<std::string>& label_strings,
                          size_t input_neurons, size_t hidden_neurons,
                          size_t output_neurons, size_t timesteps,
                          bool timeseries) {
    if (cfg.network_json_out.empty()) {
        return false;
    }

    /* Load the best saved network JSON */
    FILE* fp = fopen(cfg.network_json_out.c_str(), "r");
    if (!fp) {
        fprintf(stderr,
                "Warning: --confusion_matrix set but best network file not "
                "found at: %s\n",
                cfg.network_json_out.c_str());
        return false;
    }
    fclose(fp);

    json best_json;
    {
        ifstream ifs(cfg.network_json_out);
        ifs >> best_json;
    }
    auto* best_net = new Network();
    best_net->from_json(best_json);

    /* Create a processor for the best network */
    Processor* p = nullptr;
    load_network(&p, best_net);

    /* Choose dataset: test if present, otherwise train */
    const Dataset* eval_dataset = &test;
    if (test.data.shape[0] == 0) {
        eval_dataset = &train;
    }

    size_t num_samples = eval_dataset->data.shape[0];
    vector<int> true_labels(num_samples);
    vector<int> pred_labels(num_samples);

    /* Evaluate each sample */
    for (size_t idx = 0; idx < num_samples; idx++) {
        pred_labels[idx] = evaluate_sample(
            p, *eval_dataset, idx, hidden_neurons, output_neurons, timesteps,
            timeseries, input_neurons);
        true_labels[idx] = (int)eval_dataset->labels.data[idx];
    }

    /* Build confusion matrix */
    size_t num_classes = label_strings.size();
    vector<vector<size_t>> cm(num_classes, vector<size_t>(num_classes, 0));
    size_t correct = 0;
    for (size_t idx = 0; idx < num_samples; idx++) {
        int t  = true_labels[idx];
        int p_ = pred_labels[idx];
        if (t >= 0 && (size_t)t < num_classes && p_ >= 0 &&
            (size_t)p_ < num_classes) {
            cm[(size_t)t][(size_t)p_]++;
            if (t == p_) {
                correct++;
            }
        }
    }

    /* Compute column widths: max of label string length and cell width */
    vector<size_t> col_widths(num_classes, 0);
    for (size_t j = 0; j < num_classes; j++) {
        size_t label_len = label_strings[j].size();
        for (size_t i = 0; i < num_classes; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%zu", cm[i][j]);
            col_widths[j] = max(col_widths[j], max(label_len, strlen(buf)));
        }
    }

    /* Total width of all column data areas */
    size_t col_data_width = 0;
    for (size_t j = 0; j < num_classes; j++) {
        col_data_width += col_widths[j] + 2; /* space + content + space */
    }
    col_data_width += num_classes; /* "|" separators */

    /* Left label column width */
    size_t left_label_width = 0;
    for (size_t i = 0; i < num_classes; i++) {
        left_label_width = max(left_label_width, label_strings[i].size());
    }
    left_label_width = max(left_label_width, (size_t)strlen("True"));
    left_label_width += 1; /* 1 space padding */

    /* Build separator line to match data column widths */
    string sep(left_label_width, '-');
    sep += '|';
    for (size_t j = 0; j < num_classes; j++) {
        sep += string(col_widths[j] + 1, '-');
        sep += '|';
    }

    printf("Confusion Matrix:\n");

    /* Axis labels: "Predicted" centered above columns */
    printf(" %*s", (int)left_label_width - 1, "");
    printf("|");
    printf(" %*s", (int)(col_data_width / 2 - 4), "Predicted");
    printf("\n");

    /* Column labels + "True" axis label on the left */
    printf(" %*s", (int)left_label_width - 1, "True");
    printf("|");
    for (size_t j = 0; j < num_classes; j++) {
        printf(" ");
        printf("%s", label_strings[j].c_str());
        for (size_t k = label_strings[j].size(); k < col_widths[j]; k++) {
            printf(" ");
        }
        printf("|");
    }
    printf("\n");
    printf("%s\n", sep.c_str());

    /* Data rows */
    for (size_t i = 0; i < num_classes; i++) {
        printf(" %*s", (int)left_label_width - 1, label_strings[i].c_str());
        printf("|");
        for (size_t j = 0; j < num_classes; j++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%zu", cm[i][j]);
            printf(" %*s", (int)col_widths[j], buf);
            printf("|");
        }
        printf("\n");
    }

    double accuracy = (double)correct / (double)num_samples;
    printf("Accuracy: %.3f (%zu/%zu)\n", accuracy, correct, num_samples);

    /* Cleanup */
    delete p;
    delete best_net;
    return true;
}
