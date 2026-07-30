#include "data_utils.h"
#include "framework.hpp"
#include <cassert>

void encode_spikes(neuro::Processor* p, const Dataset* d, size_t index,
                   size_t timesteps, bool timeseries, size_t input_neurons) {
    if (timeseries) {
        // shape = [observations * input_features * timesteps]
        size_t encoding_window = timesteps / d->data.shape[2];
        assert(encoding_window > 0);

        for (size_t input = 0; input < input_neurons / 2; input++) {
            for (int column_t = 0; column_t < d->data.shape[2]; column_t++) {
                double encoding_start = column_t * encoding_window;
                double encoding_end   = encoding_start + encoding_window;

                double x =
                    d->data.data[(index * d->data.shape[1] * d->data.shape[2]) +
                                 (input * d->data.shape[2]) + column_t];
                double inv_x = 1.0 - x;

                if (x > 0.0) {
                    for (double j = encoding_start; j < encoding_end;
                         j += 1.0 / x) {
                        p->apply_spike({(int)input * 2, (double)(int)j, 1.0});
                    }
                }
                if (inv_x > 0.0) {
                    for (double j = encoding_start; j < encoding_end;
                         j += 1.0 / inv_x) {
                        p->apply_spike(
                            {(int)input * 2 + 1, (double)(int)j, 1.0});
                    }
                }
            }
        }
    } else {
        // Use shared spike raster encoder for non-timeseries
        auto raster = encode_spike_raster(
            d->data.data + index * d->data.shape[1], d->data.shape[1],
            timesteps, nullptr, nullptr, true); // already normalized

        for (size_t t = 0; t < timesteps; t++) {
            for (size_t n = 0; n < raster.size(); n++) {
                if (raster[n][t]) {
                    p->apply_spike({(int)n, (double)(size_t)t, 1.0});
                }
            }
        }
    }
}

std::vector<std::vector<bool>>
encode_spike_raster(const double* values, size_t n_features, size_t timesteps,
                    const double* min_vals, const double* max_vals,
                    bool use_normalized) {
    std::vector<std::vector<bool>> spikes(n_features * 2,
                                          std::vector<bool>(timesteps, false));

    for (size_t input = 0; input < n_features; input++) {
        double x;
        if (use_normalized) {
            x = values[input];
        } else {
            double range = max_vals[input] - min_vals[input];
            if (range == 0.0) {
                continue;
            }
            x = (values[input] - min_vals[input]) / range;
        }
        double inv_x = 1.0 - x;

        if (x > 0.0) {
            for (double j = 0.0; j < (double)timesteps; j += 1.0 / x) {
                spikes[input * 2][(size_t)j] = true;
            }
        }
        if (inv_x > 0.0) {
            for (double j = 0.0; j < (double)timesteps; j += 1.0 / inv_x) {
                spikes[input * 2 + 1][(size_t)j] = true;
            }
        }
    }

    return spikes;
}
