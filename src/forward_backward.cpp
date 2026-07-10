#include "forward_backward.h"
#include "data_utils.h"
#include "framework.hpp"
#include "math_utils.h"
#include "shared.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cfloat>

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

void backward(TrainingBundle* tb, const NetworkConfiguration* nc) {
    tb->future_mem_grad_.setZero();
    tb->sgh.setZero();
    tb->vgh.setZero();
    tb->dL_dV_.setZero();
    tb->v_pre_t_.setZero();
    tb->dV_post_dV_pre_.setZero();
    tb->dV_post_ds_t_.setZero();
    tb->ds_t_dV_pre_.setZero();
    tb->dV_leak_dV_t1_.setZero();
    tb->grad_.setZero();

    for (int t = nc->timesteps - 1; t >= 0; t--) {
        tb->sgh.col(t).segment(nc->layer_offsets[2], nc->output_neurons) +=
            Eigen::Map<const Eigen::VectorXd>(&tb->dL_ds[0],
                                              nc->output_neurons) /
            nc->timesteps;

        tb->dL_dV_   = tb->vgh.col(t) + tb->future_mem_grad_;
        tb->v_pre_t_ = Eigen::Map<const Eigen::VectorXd>(&tb->v_pre[t][0],
                                                         nc->total_neurons);

        tb->dV_post_dV_pre_ = (Eigen::Map<const Eigen::VectorXd>(
                                   &tb->spikes[t][0], nc->total_neurons)
                                   .array() <= 0)
                                  .cast<double>();

        tb->dV_post_ds_t_ = -tb->v_pre_t_;
        if (nc->min_potential > 0) {
            (tb->dV_post_ds_t_.array() + nc->min_potential).matrix();
        }

        tb->ds_t_dV_pre_ =
            (tb->rho / (2.0 * tb->tau)) *
            (-(tb->v_pre_t_ - Eigen::Map<const Eigen::VectorXd>(
                                  &((*tb->thresholds)[0]), nc->total_neurons))
                  .array()
                  .abs()
                  .matrix() /
             tb->tau)
                .array()
                .exp()
                .matrix();

        tb->dV_leak_dV_t1_ =
            (tb->v_pre_t_.array() >= nc->min_potential).cast<double>() *
            (1.0 - nc->leak);

        tb->grad_ = (tb->dL_dV_.array() * tb->dV_post_dV_pre_.array()) +
                    (tb->dL_dV_.array() * tb->dV_post_ds_t_.array() *
                     tb->ds_t_dV_pre_.array()) +
                    (tb->sgh.col(t).array() * tb->ds_t_dV_pre_.array());

        tb->future_mem_grad_ =
            (tb->dL_dV_.array() * tb->dV_post_dV_pre_.array() *
             tb->dV_leak_dV_t1_.array()) +
            (tb->dL_dV_.array() * tb->dV_post_ds_t_.array() *
             tb->ds_t_dV_pre_.array() * tb->dV_leak_dV_t1_.array()) +
            (tb->sgh.col(t).array() * tb->ds_t_dV_pre_.array() *
             tb->dV_leak_dV_t1_.array());

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
                tb->delta_W[dest][source_idx] += source_spike * tb->grad_(dest);
                tb->sgh(source, source_time) +=
                    tb->grad_(dest) * (*tb->weights)[dest][source_idx];
            }
        }
    }
}
