#include "math_utils.h"
#include <cmath>
#include <cstdlib>

double normal(double mean, double stddev) {
    double u1, u2;
    do {
        u1 = drand48();
    } while (u1 == 0.0); // Avoid log(0)
    u2       = drand48();
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * acos(-1.0) * u2);
    return mean + stddev * z;
}

void softmax(const double* logits, double* out, size_t n) {
    double max = logits[0];
    for (size_t i = 1; i < n; i++) {
        if (logits[i] > max) {
            max = logits[i];
        }
    }

    double exp_sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        out[i] = exp(logits[i] - max);
        exp_sum += out[i];
    }

    for (size_t i = 0; i < n; i++) {
        out[i] /= exp_sum;
    }
}

double cross_entropy(const double* logits, const double* targets, double* grads, size_t n) {
    softmax(logits, grads, n);

    double loss = 0.0;
    for (size_t i = 0; i < n; i++) {
        loss -= targets[i] * log(grads[i] + ADAM_EPS);
        grads[i] -= targets[i];
    }

    return loss;
}

double alpha(bool leak) { return (double)!leak; }

double spike_surrogate(double v_pre_t, double v_thresh, double scale_rho, double tau_rho_scaled) {
    return (scale_rho / (2.0 * tau_rho_scaled)) *
           expf(-fabs(v_pre_t - v_thresh) / tau_rho_scaled);
}

double quantize(double weight, int steps, int min, int max) {
    int x = round(weight / (2.0 / steps));

    if (x < min) {
        x = min;
    } else if (x > max) {
        x = max;
    }

    return x * (2.0 / steps);
}
