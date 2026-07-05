#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <cstddef>

#define ADAM_EPS (1.0e-8)

double normal(double mean, double stddev);
void softmax(const double* logits, double* out, size_t n);
double cross_entropy(const double* logits, const double* targets, double* grads,
                     size_t n);
double alpha(bool leak);
double spike_surrogate(double v_pre_t, double v_thresh, double scale_rho,
                       double tau_rho_scaled);
double quantize(double weight, int steps, int min, int max);

#endif // MATH_UTILS_H
