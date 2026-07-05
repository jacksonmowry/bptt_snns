#include "weight_updates.h"
#include "csv.h"
#include "framework.hpp"
#include "math_utils.h"
#include <algorithm>

void weight_updates(const NetworkConfiguration* nc, const Dataset* d,
                    size_t current_batch_size, size_t batch_size,
                    size_t batch_start, size_t epoch, double& b1_t,
                    double& b2_t, std::vector<std::vector<double>>& m_weights,
                    std::vector<std::vector<double>>& v_weights,
                    double learning_rate, double decay_rate,
                    std::vector<std::vector<double>>& weights,
                    std::vector<std::vector<double>>& delta_W) {
    double inv_batch = 1.0 / ((double)current_batch_size * nc->timesteps);

    b1_t *= BETA1;
    b2_t *= BETA2;

    for (size_t i = 0; i < nc->total_neurons; i++) {
        for (size_t j = 0; j < nc->n->get_node(i)->incoming.size(); j++) {
            neuro::Edge* e = nc->n->get_node(i)->incoming[j];

            delta_W[i][j] *= inv_batch;

            m_weights[i][j] =
                BETA1 * m_weights[i][j] + (1.0 - BETA1) * delta_W[i][j];
            v_weights[i][j] = BETA2 * v_weights[i][j] +
                              (1.0 - BETA2) * (delta_W[i][j] * delta_W[i][j]);
            delta_W[i][j]   = 0.0;

            double mW_hat = m_weights[i][j] / (1.0 - b1_t);
            double vW_hat = v_weights[i][j] / (1.0 - b2_t);

            double lr = learning_rate;
            if (epoch == 0) {
                lr = ((batch_start + batch_size) / (double)d->observations) *
                     learning_rate;
            }

            weights[i][j] -= lr * mW_hat / (sqrt(vW_hat + ADAM_EPS));
            weights[i][j] -= lr * decay_rate * weights[i][j];
            if (weights[i][j] > 1.0) {
                weights[i][j] = 1.0;
            } else if (weights[i][j] < -1.0) {
                weights[i][j] = -1.0;
            }

            if (nc->discrete) {
                weights[i][j] = quantize(weights[i][j], 256, -127, 127);
            }

            e->set("Weight",
                   weights[i][j] / (nc->discrete ? nc->scale_factor : 1.0));
        }
    }
}
