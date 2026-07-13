#include "data_utils.h"
#include "framework.hpp"
#include <cassert>

size_t label_count(const Dataset* d) { return (size_t)d->label_strings_count; }

void encode_spikes(neuro::Processor* p, const Dataset* d, size_t index,
                   size_t timesteps, bool timeseries, size_t input_neurons) {
    if (timeseries) {
        // shape = [observations * input_features * timesteps]
        size_t encoding_window = timesteps / d->shape[2];
        assert(encoding_window > 0);

        for (size_t input = 0; input < input_neurons / 2; input++) {
            double range = d->max_vals[input] - d->min_vals[input];

            for (int column_t = 0; column_t < d->shape[2]; column_t++) {
                double encoding_start = column_t * encoding_window;
                double encoding_end   = encoding_start + encoding_window;

                double x     = (d->data[(index * d->shape[1] * d->shape[2]) +
                                        (input * d->shape[2]) + column_t] -
                                d->min_vals[input]) /
                               range;
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
        for (size_t input = 0; input < input_neurons / 2; input++) {
            double x =
                (d->data[index * d->shape[1] + input] - d->min_vals[input]) /
                (d->max_vals[input] - d->min_vals[input]);
            double inv_x = 1.0 - x;
            if (x > 0.0) {
                for (double j = 0; j < (double)timesteps; j += 1.0 / x) {
                    p->apply_spike({(int)input * 2, (double)(size_t)j, 1.0});
                }
            }
            if (inv_x > 0.0) {
                for (double j = 0; j < (double)timesteps; j += 1.0 / inv_x) {
                    p->apply_spike(
                        {(int)input * 2 + 1, (double)(size_t)j, 1.0});
                }
            }
        }
    }
}
