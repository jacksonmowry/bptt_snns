#include "forward_backward.h"
#include "data_utils.h"
#include "framework.hpp"
#include "math_utils.h"
#include "shared.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstring>

using namespace neuro;

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

        const std::vector<int>& neuron_counts         = p->neuron_counts();
        const std::vector<double>& neuron_pre_charges = p->neuron_pre_charges();
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

    if (nc->loss_func == LossFunc::CCE) {
        /* Classification: assert single-column label (one-hot target) */
        if (d->labels.dims == 1) {
            assert(d->labels.shape[0] > 0);
        } else {
            assert(d->labels.dims == 2);
            assert(d->labels.shape[1] == 1);
        }

        double label_val = d->labels.data[index];
        size_t label_idx = (size_t)label_val;
        assert(label_idx < nc->output_neurons);

        if (max_idx == label_idx) {
            er.correct++;
        }

        for (size_t i = 0; i < nc->output_neurons; i++) {
            if (i == label_idx) {
                tb->target[i] = 1.0;
            } else {
                tb->target[i] = 0.0;
            }
        }

        double loss_spike =
            cross_entropy(tb->spike_logits.data(), tb->target.data(),
                          tb->dL_ds.data(), nc->output_neurons);
        er.loss = loss_spike;

    } else {
        // Regression (MSE): 2D (== output neurons)
        assert(d->labels.dims == 2);
        assert(d->labels.shape[0] > 0);
        assert(d->labels.shape[1] == (int)nc->output_neurons);

        // Accuracy not meaningful for MSE — leave at 0
        er.correct = 0;

        // Populate target from label row
        memset(tb->target.data(), 0, nc->output_neurons * sizeof(double));

        // 2D labels: multiple targets per observation
        const double* label_row = d->labels.data + index * nc->output_neurons;

        memcpy(tb->target.data(), label_row,
               nc->output_neurons * sizeof(double));

        double loss_spike = mse(tb->spike_logits.data(), tb->target.data(),
                                tb->dL_ds.data(), nc->output_neurons);
        er.loss           = loss_spike;
    }

    return er;
}

void backward(TrainingBundle* tb, const NetworkConfiguration* nc) {
    tb->future_mem_grad.setZero();
    tb->sgh.setZero();
    tb->dL_dV.setZero();
    tb->v_pre_t.setZero();
    tb->dV_post_dV_pre.setZero();
    tb->dV_post_ds_t.setZero();
    tb->ds_t_dV_pre.setZero();
    tb->dV_leak_dV_t1.setZero();
    tb->grad.setZero();

    for (int t = nc->timesteps - 1; t >= 0; t--) {
        tb->sgh.col(t).segment(nc->layer_offsets[2], nc->output_neurons) +=
            Eigen::Map<const Eigen::VectorXd>(&tb->dL_ds[0],
                                              nc->output_neurons) /
            nc->timesteps;

        tb->dL_dV   = tb->future_mem_grad;
        tb->v_pre_t = Eigen::Map<const Eigen::VectorXd>(&tb->v_pre[t][0],
                                                        nc->total_neurons);

        tb->dV_post_dV_pre = (Eigen::Map<const Eigen::VectorXd>(
                                  &tb->spikes[t][0], nc->total_neurons)
                                  .array() <= 0)
                                 .cast<double>();

        tb->dV_post_ds_t = -tb->v_pre_t;
        if (nc->min_potential > 0) {
            tb->dV_post_ds_t =
                (tb->dV_post_ds_t.array() + nc->min_potential).matrix();
        }

        tb->ds_t_dV_pre =
            (tb->rho / (2.0 * tb->tau)) *
            (-(tb->v_pre_t - Eigen::Map<const Eigen::VectorXd>(
                                 &((*tb->thresholds)[0]), nc->total_neurons))
                  .array()
                  .abs()
                  .matrix() /
             tb->tau)
                .array()
                .exp()
                .matrix();

        tb->dV_leak_dV_t1 =
            (tb->v_pre_t.array() >= nc->min_potential).cast<double>() *
            (1.0 - nc->leak);

        tb->grad = (tb->dL_dV.array() * tb->dV_post_dV_pre.array()) +
                   (tb->dL_dV.array() * tb->dV_post_ds_t.array() *
                    tb->ds_t_dV_pre.array()) +
                   (tb->sgh.col(t).array() * tb->ds_t_dV_pre.array());

        tb->future_mem_grad =
            (tb->dL_dV.array() * tb->dV_post_dV_pre.array() *
             tb->dV_leak_dV_t1.array()) +
            (tb->dL_dV.array() * tb->dV_post_ds_t.array() *
             tb->ds_t_dV_pre.array() * tb->dV_leak_dV_t1.array()) +
            (tb->sgh.col(t).array() * tb->ds_t_dV_pre.array() *
             tb->dV_leak_dV_t1.array());

        for (int dest = nc->total_neurons - 1; dest >= 0; dest--) {
            for (size_t source_idx = 0;
                 source_idx < nc->n->get_node(dest)->incoming.size();
                 source_idx++) {
                size_t source =
                    nc->n->get_node(dest)->incoming[source_idx]->from->id;

                int delay       = (*tb->delays)[dest][source_idx];
                int source_time = t - delay;
                if (source_time < 0) {
                    continue;
                }

                double source_spike = tb->spikes[source_time][source];
                tb->delta_W[dest][source_idx] += source_spike * tb->grad(dest);
                tb->sgh(source, source_time) +=
                    tb->grad(dest) * (*tb->weights)[dest][source_idx];
            }
        }
    }
}
