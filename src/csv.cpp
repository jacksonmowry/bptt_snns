#include "csv.h"
#include <algorithm>
#include <assert.h>
#include <cfloat>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Forward declarations for helpers used by high-level loaders
static int compute_2d_dims(FILE* f_data, int* p_cols);

// Build unique sorted label list from raw strings. Returns count.
static std::vector<std::string>
build_label_mapping(char** raw_labels, int obs_count, double* indices_out) {
    std::vector<std::string> label_strings_out;

    for (int i = 0; i < obs_count; i++) {
        bool found = false;
        for (auto& u : label_strings_out) {
            if (u == std::string(raw_labels[i])) {
                found = true;
                break;
            }
        }
        if (!found) {
            label_strings_out.push_back(std::string(raw_labels[i]));
        }
    }

    std::sort(label_strings_out.begin(), label_strings_out.end());

    for (int i = 0; i < obs_count; i++) {
        for (int j = 0; j < (int)label_strings_out.size(); j++) {
            if (label_strings_out[j] == std::string(raw_labels[i])) {
                indices_out[i] = (double)j;
                break;
            }
        }
    }

    return label_strings_out;
}

// Read all labels as strings. Returns array of strings (caller frees).
static char** read_label_strings(FILE* f, int rows) {
    char line[4096 * 16];
    char** labels = (char**)malloc(rows * sizeof(char*));
    for (int i = 0; i < rows; i++) {
        if (fgets(line, sizeof(line), f) != NULL) {
            assert(strlen(line) < sizeof(line) - 2);
            char* nl = strchr(line, '\n');
            if (nl) {
                *nl = '\0';
            }
            char* cr = strchr(line, '\r');
            if (cr) {
                *cr = '\0';
            }
            labels[i] = strdup(line);
        } else {
            labels[i] = strdup("");
        }
    }
    return labels;
}

// Compute min/max values for RealData from its data buffer.
// Uses shape/dims to determine iteration bounds.
// Caller must have allocated rd.min_vals and rd.max_vals.
void compute_realdata_minmax(RealData& rd) {
    if (!rd.data || !rd.min_vals || !rd.max_vals || !rd.shape) {
        return;
    }

    if (rd.dims == 2) {
        for (int j = 0; j < rd.shape[1]; j++) {
            rd.min_vals[j] = rd.data[j];
            rd.max_vals[j] = rd.data[j];
        }
        for (int i = 1; i < rd.shape[0]; i++) {
            for (int j = 0; j < rd.shape[1]; j++) {
                double val =
                    rd.data[(size_t)i * (size_t)rd.shape[1] + (size_t)j];
                if (val < rd.min_vals[j]) {
                    rd.min_vals[j] = val;
                }
                if (val > rd.max_vals[j]) {
                    rd.max_vals[j] = val;
                }
            }
        }
    } else if (rd.dims == 3) {
        for (int feature = 0; feature < rd.shape[1]; feature++) {
            rd.min_vals[feature] = DBL_MAX;
            rd.max_vals[feature] = -DBL_MAX;
        }
        for (int obs = 0; obs < rd.shape[0]; obs++) {
            for (int feature = 0; feature < rd.shape[1]; feature++) {
                for (int column = 0; column < rd.shape[2]; column++) {
                    size_t idx = (size_t)(obs * rd.shape[1] * rd.shape[2] +
                                          feature * rd.shape[2] + column);
                    double val = rd.data[idx];
                    if (val < rd.min_vals[feature]) {
                        rd.min_vals[feature] = val;
                    }
                    if (val > rd.max_vals[feature]) {
                        rd.max_vals[feature] = val;
                    }
                }
            }
        }
    }
}

// Normalize RealData to [0,1] range using its min_vals/max_vals.
// Values with zero range are set to NaN to match original behavior.
// Modifies rd.data in-place; min_vals/max_vals unchanged.
void normalize_realdata(RealData& rd) {
    if (!rd.data || !rd.min_vals || !rd.max_vals || !rd.shape) {
        return;
    }

    if (rd.dims == 2) {
        for (int i = 0; i < rd.shape[0]; i++) {
            for (int j = 0; j < rd.shape[1]; j++) {
                double range = rd.max_vals[j] - rd.min_vals[j];
                if (range == 0.0) {
                    rd.data[(size_t)i * (size_t)rd.shape[1] + (size_t)j] =
                        0.0 / 0.0; // NaN
                } else {
                    rd.data[(size_t)i * (size_t)rd.shape[1] + (size_t)j] =
                        (rd.data[(size_t)i * (size_t)rd.shape[1] + (size_t)j] -
                         rd.min_vals[j]) /
                        range;
                }
            }
        }
    } else if (rd.dims == 3) {
        for (int obs = 0; obs < rd.shape[0]; obs++) {
            for (int feature = 0; feature < rd.shape[1]; feature++) {
                double range = rd.max_vals[feature] - rd.min_vals[feature];
                for (int column = 0; column < rd.shape[2]; column++) {
                    size_t idx = (size_t)(obs * rd.shape[1] * rd.shape[2] +
                                          feature * rd.shape[2] + column);
                    if (range == 0.0) {
                        rd.data[idx] = 0.0 / 0.0; // NaN
                    } else {
                        rd.data[idx] =
                            (rd.data[idx] - rd.min_vals[feature]) / range;
                    }
                }
            }
        }
    }
}

// Parse 2D (non-timeseries) data from file.
// Returns RealData with data, min_vals, max_vals, shape allocated.
// Caller must free: data.data, data.min_vals, data.max_vals, data.shape.
static RealData parse_realdata_2d(FILE* f_data, int rows, int cols) {
    RealData rd = {NULL, NULL, NULL, 2, NULL};

    rd.shape    = (int*)malloc(2 * sizeof(int));
    rd.shape[0] = rows;
    rd.shape[1] = cols;

    rd.data = (double*)malloc((size_t)rows * (size_t)cols * sizeof(double));

    size_t line_len = 4096 * 16;
    char* line      = (char*)malloc(4096 * 16);
    rewind(f_data);
    for (int i = 0; i < rows; i++) {
        if (fgets(line, line_len, f_data) != NULL) {
            bool blank = true;
            for (size_t j = 0; blank && j < strlen(line); j++) {
                blank &= (bool)(isspace(line[j]));
            }

            if (blank) {
                // Skipping blank lines
                i--;
                continue;
            }

            char* token = strtok(line, ",");
            int parsed  = 0;
            for (int j = 0; j < cols && token != NULL; j++) {
                rd.data[(size_t)i * (size_t)cols + (size_t)j] = atof(token);
                token = strtok(NULL, ",");
                parsed++;
            }
            if (parsed < cols) {
                fprintf(stderr,
                        "%s: truncated line %d "
                        "(expected %d values, got %d)\n",
                        __func__, i + 1, cols, parsed);
            }
        }
    }
    free(line);

    // Min/max calcs
    compute_realdata_minmax(rd);

    return rd;
}

// Parse labels from file.
// Returns {label indices as RealData (dims=1), vector of label strings}.
// Caller must free the RealData via free_realdata().
static std::pair<RealData, std::vector<std::string>>
parse_textdata(FILE* f_labels, int count) {
    RealData rd = {NULL, NULL, NULL, 1, NULL};

    rd.data = (double*)malloc(count * sizeof(double));

    rewind(f_labels);
    char** raw_labels = read_label_strings(f_labels, count);

    std::vector<std::string> label_strings =
        build_label_mapping(raw_labels, count, rd.data);

    // free raw_labels now that we're done with them
    for (int i = 0; i < count; i++) {
        free(raw_labels[i]);
    }
    free(raw_labels);

    rd.shape    = (int*)malloc(1 * sizeof(int));
    rd.shape[0] = count;

    return {rd, std::move(label_strings)};
}

// Load 2D data from file. Returns RealData with dims=2, shape=[rows, cols].
// Caller must free via free_realdata(). Returns zero-initialized on failure.
RealData load_realdata_2d(const char* path) {
    RealData rd = {NULL, NULL, NULL, 2, NULL};
    FILE* f     = fopen(path, "r");
    if (!f) {
        return rd;
    }

    int cols;
    int rows = compute_2d_dims(f, &cols);
    fclose(f);

    if (rows == 0 || cols == 0) {
        rd.shape    = (int*)malloc(2 * sizeof(int));
        rd.shape[0] = 0;
        rd.shape[1] = 0;
        return rd;
    }

    // Re-open for parsing
    f = fopen(path, "r");
    if (!f) {
        rd.shape    = (int*)malloc(2 * sizeof(int));
        rd.shape[0] = 0;
        rd.shape[1] = 0;
        return rd;
    }

    rd = parse_realdata_2d(f, rows, cols);
    fclose(f);

    if (!rd.data) {
        free_realdata(rd);
        rd          = {NULL, NULL, NULL, 2, NULL};
        rd.shape    = (int*)malloc(2 * sizeof(int));
        rd.shape[0] = 0;
        rd.shape[1] = 0;
    }
    return rd;
}

// Load text labels from file. Returns {label indices RealData (dims=1), vector
// of label strings}. count = number of observations (rows) in the file. Caller
// must free the RealData via free_realdata().
std::pair<RealData, std::vector<std::string>> load_textdata(const char* path,
                                                            int count) {
    RealData rd = {NULL, NULL, NULL, 1, NULL};

    if (count <= 0) {
        rd.shape    = (int*)malloc(1 * sizeof(int));
        rd.shape[0] = 0;
        return {rd, {}};
    }

    FILE* f = fopen(path, "r");
    if (!f) {
        rd.shape    = (int*)malloc(1 * sizeof(int));
        rd.shape[0] = 0;
        return {rd, {}};
    }

    auto result = parse_textdata(f, count);
    fclose(f);

    if (!result.first.data) {
        free_realdata(result.first);
        result.first          = {NULL, NULL, NULL, 1, NULL};
        result.first.shape    = (int*)malloc(1 * sizeof(int));
        result.first.shape[0] = 0;
    }
    return result;
}

// Count non-blank lines and compute dimensions for 2D data.
static int compute_2d_dims(FILE* f_data, int* p_cols) {
    fprintf(stderr, "%s\n", __FUNCTION__);
    size_t line_len = 4096 * 16;
    char* line      = (char*)malloc(4096 * 16);
    int rows        = 0;

    while (fgets(line, line_len, f_data) != NULL) {
        // Skip blank lines
        bool blank = true;
        for (size_t i = 0; blank && i < strlen(line); i++) {
            blank &= (bool)(isspace(line[i]));
        }

        if (!blank) {
            rows++;
        }
    }

    rewind(f_data);
    int cols = 0;
    if (fgets(line, line_len, f_data) != NULL) {
        for (int i = 0; line[i] != '\0' && line[i] != '\n'; i++) {
            if (line[i] == ',') {
                cols++;
            }
        }
        cols++;
    }

    free(line);
    *p_cols = cols;
    fprintf(stderr, "Rows: %d, cols: %d\n", rows, cols);
    return rows;
}

// Compute the stride between consecutive entries (in doubles) from shape/dims.
// stride = product of shape[1..dims-1]. For 1D data (dims=1), stride=1.
static size_t entry_stride_doubles(const int* shape, int dims) {
    size_t stride = 1;
    for (int i = 1; i < dims; i++) {
        stride *= (size_t)shape[i];
    }
    return stride;
}

// Shuffle two double arrays simultaneously with the same permutation.
// Operates on double* data with dims/shape to compute entry stride.
// Uses rand() % i for i = 1..num_entries-1, matching the original 2D/3D
// implementations.
static void shuffle_two(double* data_a, const int* shape_a, int dims_a,
                        double* data_b, const int* shape_b, int dims_b) {
    size_t num_entries = (size_t)shape_a[0];
    size_t stride_a    = entry_stride_doubles(shape_a, dims_a);
    size_t stride_b    = entry_stride_doubles(shape_b, dims_b);

    double* tmp_a = (double*)malloc(stride_a * sizeof(double));
    double* tmp_b = (double*)malloc(stride_b * sizeof(double));
    if (!tmp_a || !tmp_b) {
        free(tmp_a);
        free(tmp_b);
        return;
    }

    for (size_t i = 1; i < num_entries; i++) {
        size_t swap_idx = rand() % i;
        memcpy(tmp_a, data_a + (i * stride_a), stride_a * sizeof(double));
        memcpy(data_a + (i * stride_a), data_a + (swap_idx * stride_a),
               stride_a * sizeof(double));
        memcpy(data_a + (swap_idx * stride_a), tmp_a,
               stride_a * sizeof(double));
        memcpy(tmp_b, data_b + (i * stride_b), stride_b * sizeof(double));
        memcpy(data_b + (i * stride_b), data_b + (swap_idx * stride_b),
               stride_b * sizeof(double));
        memcpy(data_b + (swap_idx * stride_b), tmp_b,
               stride_b * sizeof(double));
    }
    free(tmp_a);
    free(tmp_b);
}

// Free all members of a RealData.
void free_realdata(RealData rd) {
    free(rd.data);
    free(rd.min_vals);
    free(rd.max_vals);
    free(rd.shape);
}

// Helper: create an empty Dataset
static Dataset make_empty_dataset() {
    Dataset out;
    out.data       = {NULL, NULL, NULL, 0, NULL};
    out.labels     = {NULL, NULL, NULL, 0, NULL};
    out.timeseries = 0;
    return out;
}

// Generic split builder: handles both 2D and 3D, both classification and
// regression.
// - data: input features RealData (source)
// - labels: label/target RealData (source),
// - start, len: range within source
// - out: output Dataset (filled)
static void build_split_dataset(const RealData* data, const RealData* labels,
                                int start, int len, Dataset* out) {
    *out = make_empty_dataset();

    size_t stride = entry_stride_doubles(data->shape, data->dims);

    // Allocate data
    out->data.data   = (double*)malloc((size_t)len * stride * sizeof(double));
    out->labels.data = (double*)malloc(
        (size_t)len * entry_stride_doubles(labels->shape, labels->dims) *
        sizeof(double));
    out->data.min_vals = (double*)malloc(data->shape[1] * sizeof(double));
    out->data.max_vals = (double*)malloc(data->shape[1] * sizeof(double));

    if (!out->data.data || !out->labels.data || !out->data.min_vals ||
        !out->data.max_vals) {
        free(out->data.data);
        free(out->labels.data);
        free(out->data.min_vals);
        free(out->data.max_vals);
        *out = make_empty_dataset();
        return;
    }

    // For regression labels: allocate min/max; for classification labels no
    // min/max needed
    if (labels->dims > 1) {
        out->labels.min_vals =
            (double*)malloc(labels->shape[1] * sizeof(double));
        out->labels.max_vals =
            (double*)malloc(labels->shape[1] * sizeof(double));
        if (!out->labels.min_vals || !out->labels.max_vals) {
            free(out->labels.min_vals);
            free(out->labels.max_vals);
            free(out->data.data);
            free(out->labels.data);
            free(out->data.min_vals);
            free(out->data.max_vals);
            *out = make_empty_dataset();
            return;
        }
    }

    // Copy data subset
    memcpy(out->data.data, data->data + (size_t)start * stride,
           (size_t)len * stride * sizeof(double));
    // Copy labels subset
    size_t label_stride = entry_stride_doubles(labels->shape, labels->dims);
    memcpy(out->labels.data, labels->data + (size_t)start * label_stride,
           (size_t)len * label_stride * sizeof(double));

    // For regression: copy min/max for labels
    if (labels->dims > 1) {
        assert(out->labels.min_vals);
        assert(labels->min_vals);
        assert(out->labels.max_vals);
        assert(labels->max_vals);
        memcpy(out->labels.min_vals, labels->min_vals,
               labels->shape[1] * sizeof(double));
        memcpy(out->labels.max_vals, labels->max_vals,
               labels->shape[1] * sizeof(double));
    }

    // Set shape info
    out->data.dims  = data->dims;
    out->data.shape = (int*)malloc(data->dims * sizeof(int));
    memcpy(out->data.shape, data->shape, data->dims * sizeof(int));
    out->data.shape[0] = len;

    out->labels.dims  = labels->dims;
    out->labels.shape = (int*)malloc(labels->dims * sizeof(int));
    memcpy(out->labels.shape, labels->shape, labels->dims * sizeof(int));
    out->labels.shape[0] = len;

    out->timeseries = (data->dims == 3);

    // Recompute min/max on split data subset
    compute_realdata_minmax(out->data);

    // Normalize split data to [0,1] using split's min/max
    normalize_realdata(out->data);

    // For regression labels (dims > 1): compute and normalize
    // Classification labels (dims=1) are integer indices, no normalization
    // needed
    if (labels->dims > 1) {
        compute_realdata_minmax(out->labels);
        normalize_realdata(out->labels);
    }
}

// Free a Dataset. All heap memory is freed.
void free_dataset(Dataset* ds) {
    if (!ds) {
        return;
    }
    free_realdata(ds->data);
    free_realdata(ds->labels);
    ds->data.data       = NULL;
    ds->data.min_vals   = NULL;
    ds->data.max_vals   = NULL;
    ds->data.shape      = NULL;
    ds->labels.data     = NULL;
    ds->labels.min_vals = NULL;
    ds->labels.max_vals = NULL;
    ds->labels.shape    = NULL;
}

// Load a single dataset (no shuffle, no split). Detects 2D vs 3D
// via timeseries flag. Handles both classification and regression.
// For classification: labels loaded as text strings mapped to indices.
// For regression: labels loaded as numeric RealData.
// Caller must free via free_dataset().
std::pair<Dataset, std::vector<std::string>>
load_dataset_single(const char* data_path, const char* labels_path,
                    size_t timeseries, bool is_regression) {
    fprintf(stderr, "%s\n", __FUNCTION__);
    Dataset out = make_empty_dataset();
    std::vector<std::string> label_strings;

    RealData rd;
    RealData label_rd;

    rd = load_realdata_2d(data_path);
    if (timeseries) {
        assert(rd.dims == 2);
        int current_len = rd.shape[0] * rd.shape[1];
        int cols        = rd.shape[1];

        int new_rows = rd.shape[0] / timeseries;
        int new_len  = new_rows * timeseries * rd.shape[1];

        if (current_len != new_len) {
            fprintf(stderr,
                    "Attempting to reshape [%d, %d] to [%d, %d, %d] failed. "
                    "%d != %d\n",
                    rd.shape[0], rd.shape[1], new_rows, (int)timeseries,
                    rd.shape[1], current_len, new_len);
            exit(1);
        }

        rd.dims  = 3;
        rd.shape = (int*)realloc(rd.shape, rd.dims * sizeof(*rd.shape));
        if (!rd.shape) {
            perror("realloc");
            exit(1);
        }

        rd.shape[0] = new_rows;
        rd.shape[1] = (int)timeseries;
        rd.shape[2] = cols;
    }

    // For 2D this is #columns, for 3D this is #features
    rd.min_vals = (double*)malloc(rd.shape[1] * sizeof(*rd.min_vals));
    rd.max_vals = (double*)malloc(rd.shape[1] * sizeof(*rd.max_vals));
    assert(rd.min_vals);
    assert(rd.max_vals);

    assert(rd.shape);
    int count = rd.shape[0];

    if (is_regression) {
        // Regression: labels are numeric RealData.
        // Always load as 2D — timeseries flag only affects input data
        // structure, not label structure. Regression targets are
        // per-observation (one vector per sample), not sequences.
        label_rd = load_realdata_2d(labels_path);
        label_rd.min_vals =
            (double*)malloc(label_rd.shape[1] * sizeof(*label_rd.min_vals));
        label_rd.max_vals =
            (double*)malloc(label_rd.shape[1] * sizeof(*label_rd.max_vals));
        label_strings = {};
    } else {
        // Classification: labels are text strings mapped to integer indices
        auto label_result = load_textdata(labels_path, count);
        label_rd          = std::move(label_result.first);
        label_strings     = std::move(label_result.second);
    }

    if (rd.data && label_rd.data) {
        out.data       = rd;
        out.labels     = label_rd;
        out.timeseries = timeseries;
    } else {
        assert(false);
        free_realdata(rd);
        free_realdata(label_rd);
    }

    return {out, std::move(label_strings)};
}

// Shuffle and split a loaded dataset. Shuffles in-place, splits into
// train/test. Handles both classification and regression via dims check.
// Caller must free via free_dataset().
void split_dataset(Dataset* source, double train_percent, Dataset* train,
                   Dataset* test) {
    *train = make_empty_dataset();
    *test  = make_empty_dataset();
    assert(train_percent >= 0.00 && train_percent <= 1.00);

    if (!source->data.data || !source->labels.data) {
        return; // empty source
    }

    // Shuffle in-place (both data and labels together)
    shuffle_two(source->data.data, source->data.shape, source->data.dims,
                source->labels.data, source->labels.shape, source->labels.dims);

    int num_obs   = source->data.shape[0];
    int train_len = (int)(train_percent * num_obs);
    int test_len  = num_obs - train_len;

    build_split_dataset(&source->data, &source->labels, 0, train_len, train);
    build_split_dataset(&source->data, &source->labels, train_len, test_len,
                        test);
}

// Full dataset loading: load + shuffle + split into train/test.
// Convenience wrapper around load_dataset_single + split_dataset.
// label_strings filled with unique label names (empty for regression).
// For classification: labels_path contains text labels.
// For regression: labels_path contains numeric target data.
void load_dataset(const char* data_path, const char* labels_path,
                  double train_percent, size_t timeseries, Dataset* train,
                  Dataset* test, std::vector<std::string>& label_strings,
                  bool is_regression) {
    auto result =
        load_dataset_single(data_path, labels_path, timeseries, is_regression);
    split_dataset(&result.first, train_percent, train, test);
    free_dataset(&result.first);
    label_strings = std::move(result.second);
}
