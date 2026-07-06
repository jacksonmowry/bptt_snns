#include "data_utils.h"
#include "framework.hpp"
#include <unordered_set>
#include <cassert>

size_t label_count(const Dataset* d) {
    std::unordered_set<double> us;
    for (int i = 0; i < d->observations; i++) {
        us.insert(d->labels[i]);
    }

    return us.size();
}

void encode_spikes(neuro::Processor* p, const Dataset* d, size_t index,
                   size_t timesteps, bool timeseries, size_t input_neurons) {
    if (timeseries) {
        size_t encoding_window = timesteps / d->cols;
        assert(encoding_window > 0);

        for (size_t input = 0; input < input_neurons / 2; input++) {
            for (int column_t = 0; column_t < d->cols; column_t++) {
                double encoding_start = column_t * encoding_window;
                double encoding_end   = encoding_start + encoding_window;

                double x =
                    (d->data[(index * d->rows_per_observation * d->cols) +
                             (input * d->cols) + column_t] -
                     d->min_vals[input]) /
                    (d->max_vals[input] - d->min_vals[input]);
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
            double x = (d->data[index * d->cols + input] - d->min_vals[input]) /
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
