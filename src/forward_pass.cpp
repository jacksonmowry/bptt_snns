#include "forward_pass.h"
#include "math_utils.h"
#include "data_utils.h"
#include <algorithm>
#include <cassert>

EvaluationResults forward(TrainingBundle* tb, Processor* p, const Dataset* d,
                          size_t index, const NetworkConfiguration* nc) {
    EvaluationResults er = {0.0, 0.0};

    p->clear_activity();

    for (size_t t = 0; t < nc->timesteps; t++) {
        fill(tb->spikes[t].begin(), tb->spikes[t].end(), 0.0);
        fill(tb->v_pre[t].begin(), tb->v_pre[t].end(), 0.0);
    }
    fill(tb->spike_logits.begin(), tb->spike_logits.end(), 0.0);

    encode_spikes(p, d, index, nc->timesteps, nc->timeseries,
                  nc->input_neurons);

    for (size_t t = 0; t < nc->timesteps; t++) {
        p->run(1);

        const vector<int>& neuron_counts         = p->neuron_counts();
        const vector<double>& neuron_pre_charges = p->neuron_pre_charges();
        for (size_t neuron = 0; neuron < nc->total_neurons; neuron++) {
            tb->spikes[t][neuron] = neuron_counts[neuron];
            tb->v_pre[t][neuron]  = (neuron_pre_charges[neuron] *
                                     (nc->discrete ? nc->scale_factor : 1.0));

            if (neuron >= nc->layer_offsets[2]) {
                tb->spike_logits[neuron - nc->layer_offsets[2]] +=
                    neuron_counts[neuron];
            }
        }
    }

    size_t max_idx = 0;
    double max_val = 0;
    for (size_t neuron = 0; neuron < nc->output_neurons; neuron++) {
        tb->spike_logits[neuron] /= (double)nc->timesteps;

        if (tb->spike_logits[neuron] > max_val) {
            max_idx = neuron;
            max_val = tb->spike_logits[neuron];
        }
    }

    if (max_idx == (size_t)d->labels[index]) {
        er.correct++;
    }

    for (size_t i = 0; i < nc->output_neurons; i++) {
        if (i == (size_t)d->labels[index]) {
            tb->target[i] = 1.0;
        } else {
            tb->target[i] = 0.0;
        }
    }

    double loss_spike =
        cross_entropy(tb->spike_logits.data(), tb->target.data(),
                      tb->dL_ds.data(), nc->output_neurons);
    er.loss = loss_spike;

    return er;
}
