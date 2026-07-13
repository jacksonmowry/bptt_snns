#include "training.h"
#include <cstdio>
#include <cstdlib>
#include <nlohmann/json.hpp>

using namespace std;
using nlohmann::json;

TrainingState* init_training(const NetworkConfiguration& nc,
                             const Dataset& train, size_t threads, double rho,
                             double tau) {
    TrainingState* state = new TrainingState();
    size_t total_neurons = nc.total_neurons;

    // Init Adam buffers + delta_W
    state->m_weights.resize(total_neurons);
    state->v_weights.resize(total_neurons);
    state->delta_W.resize(total_neurons);

    for (size_t i = 0; i < total_neurons; i++) {
        size_t inc = nc.n->get_node(i)->incoming.size();
        state->m_weights[i].resize(inc);
        state->v_weights[i].resize(inc);
        state->delta_W[i].resize(inc);
    }

    // Threading
    state->tas         = (ThreadArgs*)calloc(threads, sizeof(*state->tas));
    state->tids        = (pthread_t*)calloc(threads, sizeof(*state->tids));
    state->batch_order = (size_t*)calloc(train.shape[0], sizeof(size_t));

    for (size_t i = 0; i < threads; i++) {
        state->tas[i] = ThreadArgs(
            total_neurons, nc.timesteps, nc.output_neurons, rho, tau,
            &state->weights, &state->delays, &state->thresholds,
            nullptr,                              // nc set later
            state->batch_order, nullptr, nullptr, // train/test set later
            &state->max_idx, &state->work_idx, &state->done_count, &state->mut,
            &state->have_work, &state->done_work, &state->train_p, &state->die);

        for (size_t neuron = 0; neuron < total_neurons; neuron++) {
            state->tas[i].tb.delta_W[neuron].resize(
                nc.n->get_node(neuron)->incoming.size());
        }
    }

    return state;
}
