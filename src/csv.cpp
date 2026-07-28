#include "csv.h"
#include <assert.h>
#include <cfloat>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

/* Forward declarations for helpers used by high-level loaders */
static int count_2d_lines(FILE* f_data, int* p_cols);
static void compute_3d_dims(FILE* f_data, int* p_num_obs, int* p_features, int* p_timesteps);

static int cmpstringp(const void* p1, const void* p2) {
    return strcmp(*(const char**)p1, *(const char**)p2);
}

// Build unique sorted label list from raw strings. Returns count.
static void build_label_mapping(char** raw_labels, int obs_count,
                                double* indices_out, std::vector<std::string>& label_strings_out) {
    std::vector<char*> unique;

    for (int i = 0; i < obs_count; i++) {
        bool found = false;
        for (auto* u : unique) {
            if (strcmp(raw_labels[i], u) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            unique.push_back(raw_labels[i]);
        }
    }

    qsort(unique.data(), unique.size(), sizeof(char*), cmpstringp);

    for (int i = 0; i < obs_count; i++) {
        for (int j = 0; j < (int)unique.size(); j++) {
            if (strcmp(raw_labels[i], unique[j]) == 0) {
                indices_out[i] = (double)j;
                break;
            }
        }
    }

    for (int i = 0; i < obs_count; i++) {
        bool is_unique = false;
        for (auto* u : unique) {
            if (raw_labels[i] == u) {
                is_unique = true;
                break;
            }
        }
        if (!is_unique) {
            free(raw_labels[i]);
        }
    }
    free(raw_labels);

    for (auto* s : unique) {
        label_strings_out.push_back(s);
    }
}

// Read all labels as strings. Returns array of strings (caller frees).
static char** read_label_strings(FILE* f, int rows) {
    char line[4096 * 16];
    char** labels = (char**)malloc(rows * sizeof(char*));
    for (int i = 0; i < rows; i++) {
        if (fgets(line, sizeof(line), f) != NULL) {
            assert(strlen(line) < sizeof(line) - 2);
            char* nl = strchr(line, '\n');
            if (nl) { *nl = '\0'; }
            char* cr = strchr(line, '\r');
            if (cr) { *cr = '\0'; }
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
    if (!rd.data || !rd.min_vals || !rd.max_vals || !rd.shape)
        return;

    if (rd.dims == 2) {
        for (int j = 0; j < rd.shape[1]; j++) {
            rd.min_vals[j] = rd.data[j];
            rd.max_vals[j] = rd.data[j];
        }
        for (int i = 1; i < rd.shape[0]; i++) {
            for (int j = 0; j < rd.shape[1]; j++) {
                double val = rd.data[(size_t)i * (size_t)rd.shape[1] + (size_t)j];
                if (val < rd.min_vals[j])
                    rd.min_vals[j] = val;
                if (val > rd.max_vals[j])
                    rd.max_vals[j] = val;
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
                    size_t idx = (size_t)(obs * rd.shape[1] * rd.shape[2] + feature * rd.shape[2] + column);
                    double val = rd.data[idx];
                    if (val < rd.min_vals[feature])
                        rd.min_vals[feature] = val;
                    if (val > rd.max_vals[feature])
                        rd.max_vals[feature] = val;
                }
            }
        }
    }
}

// Normalize RealData to [0,1] range using its min_vals/max_vals.
// Values with zero range are set to NaN to match original behavior.
// Modifies rd.data in-place; min_vals/max_vals unchanged.
void normalize_realdata(RealData& rd) {
    if (!rd.data || !rd.min_vals || !rd.max_vals || !rd.shape)
        return;

    if (rd.dims == 2) {
        for (int i = 0; i < rd.shape[0]; i++) {
            for (int j = 0; j < rd.shape[1]; j++) {
                double range = rd.max_vals[j] - rd.min_vals[j];
                if (range == 0.0) {
                    rd.data[(size_t)i * (size_t)rd.shape[1] + (size_t)j] = 0.0 / 0.0; // NaN
                } else {
                    rd.data[(size_t)i * (size_t)rd.shape[1] + (size_t)j] =
                        (rd.data[(size_t)i * (size_t)rd.shape[1] + (size_t)j] - rd.min_vals[j]) / range;
                }
            }
        }
    } else if (rd.dims == 3) {
        for (int obs = 0; obs < rd.shape[0]; obs++) {
            for (int feature = 0; feature < rd.shape[1]; feature++) {
                double range = rd.max_vals[feature] - rd.min_vals[feature];
                for (int column = 0; column < rd.shape[2]; column++) {
                    size_t idx = (size_t)(obs * rd.shape[1] * rd.shape[2] + feature * rd.shape[2] + column);
                    if (range == 0.0) {
                        rd.data[idx] = 0.0 / 0.0; // NaN
                    } else {
                        rd.data[idx] = (rd.data[idx] - rd.min_vals[feature]) / range;
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

    rd.data     = (double*)malloc((size_t)rows * (size_t)cols * sizeof(double));
    rd.min_vals = (double*)malloc(cols * sizeof(double));
    rd.max_vals = (double*)malloc(cols * sizeof(double));

    char line[4096 * 16];
    rewind(f_data);
    for (int i = 0; i < rows; i++) {
        if (fgets(line, sizeof(line), f_data) != NULL) {
            char* token = strtok(line, ",");
            for (int j = 0; j < cols && token != NULL; j++) {
                rd.data[(size_t)i * (size_t)cols + (size_t)j] = atof(token);
                token = strtok(NULL, ",");
            }
        }
    }

    // Min/max calcs
    compute_realdata_minmax(rd);

    return rd;
}

// Parse 3D (timeseries) data from file.
// Returns RealData with data, min_vals, max_vals, shape allocated.
// Caller must free: data.data, data.min_vals, data.max_vals, data.shape.
static RealData parse_realdata_3d(FILE* f_data, int num_obs, int input_features, int timesteps) {
    RealData rd = {NULL, NULL, NULL, 3, NULL};
    
    rd.shape    = (int*)malloc(3 * sizeof(int));
    rd.shape[0] = num_obs;
    rd.shape[1] = input_features;
    rd.shape[2] = timesteps;

    int block_size = input_features * timesteps;
    int total_data = num_obs * block_size;
    rd.data        = (double*)malloc(total_data * sizeof(double));
    rd.min_vals    = (double*)malloc(input_features * sizeof(double));
    rd.max_vals    = (double*)malloc(input_features * sizeof(double));

    char line[4096 * 16];
    rewind(f_data);
    int obs_idx  = 0;
    int line_cnt = 0;
    while (fgets(line, sizeof(line), f_data) != NULL) {
        int is_blank = 1;
        for (int k = 0; line[k]; k++) {
            if (line[k] != '\n' && line[k] != '\r' && line[k] != ' ') {
                is_blank = 0;
                break;
            }
        }
        if (is_blank) {
            obs_idx++;
            line_cnt = 0;
            continue;
        }
        char* token = strtok(line, ",");
        for (int c = 0; c < timesteps && token != NULL; c++) {
            rd.data[(obs_idx * block_size) + (line_cnt * timesteps) + c] =
                atof(token);
            token = strtok(NULL, ",");
        }
        line_cnt++;
    }

    // Min/max
    compute_realdata_minmax(rd);

    return rd;
}

// Parse labels from file.
// Returns {label indices as RealData (dims=1), vector of label strings}.
// Caller must free the RealData via free_realdata().
static std::pair<RealData, std::vector<std::string>> parse_textdata(FILE* f_labels, int count) {
    RealData rd = {NULL, NULL, NULL, 1, NULL};
    
    rd.data     = (double*)malloc(count * sizeof(double));
    
    rewind(f_labels);
    char** raw_labels = read_label_strings(f_labels, count);
    
    std::vector<std::string> label_strings;
    build_label_mapping(raw_labels, count, rd.data, label_strings);
    
    rd.shape   = (int*)malloc(1 * sizeof(int));
    rd.shape[0] = count;
    
    return {rd, std::move(label_strings)};
}

/* Load 2D data from file. Returns RealData with dims=2, shape=[rows, cols].
 * Caller must free via free_realdata(). Returns zero-initialized on failure. */
RealData load_realdata_2d(const char* path) {
    RealData rd = {NULL, NULL, NULL, 2, NULL};
    FILE* f = fopen(path, "r");
    if (!f) return rd;

    int cols;
    int rows = count_2d_lines(f, &cols);
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
        rd = {NULL, NULL, NULL, 2, NULL};
        rd.shape    = (int*)malloc(2 * sizeof(int));
        rd.shape[0] = 0;
        rd.shape[1] = 0;
    }
    return rd;
}

/* Load 3D data from file. Returns RealData with dims=3, shape=[obs, features, timesteps].
 * Caller must free via free_realdata(). Returns zero-initialized on failure. */
RealData load_realdata_3d(const char* path) {
    RealData rd = {NULL, NULL, NULL, 3, NULL};
    FILE* f = fopen(path, "r");
    if (!f) return rd;

    int num_obs, input_features, timesteps;
    compute_3d_dims(f, &num_obs, &input_features, &timesteps);
    fclose(f);

    if (num_obs == 0) {
        rd.shape    = (int*)malloc(3 * sizeof(int));
        rd.shape[0] = 0;
        rd.shape[1] = 0;
        rd.shape[2] = 0;
        return rd;
    }

    // Re-open for parsing
    f = fopen(path, "r");
    if (!f) {
        rd.shape    = (int*)malloc(3 * sizeof(int));
        rd.shape[0] = 0;
        rd.shape[1] = 0;
        rd.shape[2] = 0;
        return rd;
    }

    rd = parse_realdata_3d(f, num_obs, input_features, timesteps);
    fclose(f);

    if (!rd.data) {
        free_realdata(rd);
        rd = {NULL, NULL, NULL, 3, NULL};
        rd.shape    = (int*)malloc(3 * sizeof(int));
        rd.shape[0] = 0;
        rd.shape[1] = 0;
        rd.shape[2] = 0;
    }
    return rd;
}

/* Load text labels from file. Returns {label indices RealData (dims=1), vector of label strings}.
 * count = number of observations (rows) in the file.
 * Caller must free the RealData via free_realdata(). */
std::pair<RealData, std::vector<std::string>> load_textdata(const char* path, int count) {
    RealData rd = {NULL, NULL, NULL, 1, NULL};
    std::vector<std::string> label_strings;

    if (count <= 0) {
        rd.shape = (int*)malloc(1 * sizeof(int));
        rd.shape[0] = 0;
        return {rd, std::move(label_strings)};
    }

    FILE* f = fopen(path, "r");
    if (!f) {
        rd.shape = (int*)malloc(1 * sizeof(int));
        rd.shape[0] = 0;
        return {rd, std::move(label_strings)};
    }

    auto result = parse_textdata(f, count);
    fclose(f);

    if (!result.first.data) {
        free_realdata(result.first);
        result.first = {NULL, NULL, NULL, 1, NULL};
        result.first.shape = (int*)malloc(1 * sizeof(int));
        result.first.shape[0] = 0;
    }
    return result;
}

// Count non-blank lines and compute dimensions for 2D data.
static int count_2d_lines(FILE* f_data, int* p_cols) {
    char line[4096 * 16];
    int rows = 0;
    
    while (fgets(line, sizeof(line), f_data) != NULL) {
        rows++;
    }
    
    rewind(f_data);
    int cols = 0;
    if (fgets(line, sizeof(line), f_data) != NULL) {
        for (int i = 0; line[i] != '\0' && line[i] != '\n'; i++) {
            if (line[i] == ',') {
                cols++;
            }
        }
        cols++;
    }
    
    *p_cols = cols;
    return rows;
}

// Count observations/features/timesteps for 3D data.
static void compute_3d_dims(FILE* f_data, int* p_num_obs, int* p_features, int* p_timesteps) {
    char line[4096 * 16];
    int num_obs   = 0;
    int non_empty = 0;
    
    rewind(f_data);
    while (fgets(line, sizeof(line), f_data) != NULL) {
        int is_blank = 1;
        for (int k = 0; line[k]; k++) {
            if (line[k] != '\n' && line[k] != '\r' && line[k] != ' ') {
                is_blank = 0;
                break;
            }
        }
        if (!is_blank) {
            non_empty++;
        } else {
            num_obs++;
        }
    }
    if (non_empty > 0) {
        num_obs++;
    }
    
    rewind(f_data);
    int timesteps = 0;
    while (fgets(line, sizeof(line), f_data) != NULL) {
        int is_blank = 1;
        for (int k = 0; line[k]; k++) {
            if (line[k] != '\n' && line[k] != '\r' && line[k] != ' ') {
                is_blank = 0;
                break;
            }
        }
        if (!is_blank) {
            for (int i = 0; line[i]; i++) {
                if (line[i] == ',') {
                    timesteps++;
                }
            }
            timesteps++;
            break;
        }
    }
    
    *p_num_obs   = num_obs;
    *p_features  = (num_obs > 0) ? non_empty / num_obs : 0;
    *p_timesteps = timesteps;
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
// Uses rand() % i for i = 1..num_entries-1, matching the original 2D/3D implementations.
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
        memcpy(data_a + (swap_idx * stride_a), tmp_a, stride_a * sizeof(double));
        memcpy(tmp_b, data_b + (i * stride_b), stride_b * sizeof(double));
        memcpy(data_b + (i * stride_b), data_b + (swap_idx * stride_b),
               stride_b * sizeof(double));
        memcpy(data_b + (swap_idx * stride_b), tmp_b, stride_b * sizeof(double));
    }
    free(tmp_a);
    free(tmp_b);
}

// Shuffle a single double array in place.
// Operates on double* data with dims/shape to compute entry stride.
static void shuffle_one(double* data, const int* shape, int dims) {
    size_t num_entries = (size_t)shape[0];
    size_t stride      = entry_stride_doubles(shape, dims);

    double* tmp = (double*)malloc(stride * sizeof(double));
    if (!tmp) return;

    for (size_t i = 1; i < num_entries; i++) {
        size_t swap_idx = rand() % i;
        memcpy(tmp, data + (i * stride), stride * sizeof(double));
        memcpy(data + (i * stride), data + (swap_idx * stride),
               stride * sizeof(double));
        memcpy(data + (swap_idx * stride), tmp, stride * sizeof(double));
    }
    free(tmp);
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
    out.data = {NULL, NULL, NULL, 0, NULL};
    out.labels = {NULL, NULL, NULL, 0, NULL};
    out.timeseries = false;
    return out;
}

// Helper: allocate and populate a Dataset split (2D).
// Copies data subset, label indices, min/max; recomputes min/max and normalizes data.
static void build_split_dataset_2d(const RealData* rd, const RealData* labels,
                                    int start, int len, int cols,
                                    Dataset* out) {
    out->data.data     = (double*)malloc((size_t)len * (size_t)cols * sizeof(double));
    out->labels.data   = (double*)malloc(len * sizeof(double));
    out->data.min_vals = (double*)malloc(cols * sizeof(double));
    out->data.max_vals = (double*)malloc(cols * sizeof(double));
    if (!out->data.data || !out->labels.data || !out->data.min_vals || !out->data.max_vals) {
        free(out->data.data); free(out->labels.data);
        free(out->data.min_vals); free(out->data.max_vals);
        *out = make_empty_dataset();
        return;
    }

    out->data.dims                  = 2;
    out->data.shape                 = (int*)malloc(2 * sizeof(int));
    out->data.shape[0]              = len;
    out->data.shape[1]              = cols;
    out->labels.dims                = 1;
    out->labels.shape               = (int*)malloc(1 * sizeof(int));
    out->labels.shape[0]            = len;
    out->timeseries                 = false;

    // Copy data subset
    memcpy(out->data.data, rd->data + (start * cols),
           len * cols * sizeof(double));
    memcpy(out->labels.data, labels->data + start,
           len * sizeof(double));

    // Copy min/max from source
    memcpy(out->data.min_vals, rd->min_vals, cols * sizeof(double));
    memcpy(out->data.max_vals, rd->max_vals, cols * sizeof(double));

    // Recompute min/max on split subset
    compute_realdata_minmax(out->data);

    // Normalize split data to [0,1] using split's min/max
    normalize_realdata(out->data);
}

// Helper: allocate and populate a Dataset split (3D).
// Copies data subset, label indices, min/max; recomputes min/max and normalizes data.
static void build_split_dataset_3d(const RealData* rd, const RealData* labels,
                                    int start, int len, int block_size,
                                    int input_features,
                                    Dataset* out) {
    out->data.data     = (double*)malloc(len * block_size * sizeof(double));
    out->labels.data   = (double*)malloc(len * sizeof(double));
    out->data.min_vals = (double*)malloc(input_features * sizeof(double));
    out->data.max_vals = (double*)malloc(input_features * sizeof(double));
    if (!out->data.data || !out->labels.data || !out->data.min_vals || !out->data.max_vals) {
        free(out->data.data); free(out->labels.data);
        free(out->data.min_vals); free(out->data.max_vals);
        *out = make_empty_dataset();
        return;
    }

    out->data.dims                  = 3;
    out->data.shape                 = (int*)malloc(3 * sizeof(int));
    out->data.shape[0]              = len;
    out->data.shape[1]              = input_features;
    out->data.shape[2]              = rd->shape[2]; // timesteps
    out->labels.dims                = 1;
    out->labels.shape               = (int*)malloc(1 * sizeof(int));
    out->labels.shape[0]            = len;
    out->timeseries                 = true;

    memcpy(out->data.data, rd->data + (start * block_size),
           len * block_size * sizeof(double));
    memcpy(out->labels.data, labels->data + start,
           len * sizeof(double));
    memcpy(out->data.min_vals, rd->min_vals, input_features * sizeof(double));
    memcpy(out->data.max_vals, rd->max_vals, input_features * sizeof(double));

    // Recompute min/max on split subset
    compute_realdata_minmax(out->data);

    // Normalize split data to [0,1] using split's min/max
    normalize_realdata(out->data);
}

// Free a Dataset. All heap memory is freed.
void free_dataset(Dataset* ds) {
    if (!ds) return;
    free_realdata(ds->data);
    free_realdata(ds->labels);
    ds->data.data = NULL;
    ds->data.min_vals = NULL;
    ds->data.max_vals = NULL;
    ds->data.shape = NULL;
    ds->labels.data = NULL;
    ds->labels.min_vals = NULL;
    ds->labels.max_vals = NULL;
    ds->labels.shape = NULL;
}

/* Load a single dataset (no shuffle, no split). Detects 2D vs 3D
 * via timeseries flag. Caller must free via free_dataset(). */
std::pair<Dataset, std::vector<std::string>> load_dataset_single(
    const char* data_path, const char* labels_path, bool timeseries) {
    Dataset out = make_empty_dataset();
    std::vector<std::string> label_strings;

    RealData rd;
    RealData label_rd;

    if (timeseries) {
        rd = load_realdata_3d(data_path);
    } else {
        rd = load_realdata_2d(data_path);
    }

    int count = rd.shape ? rd.shape[0] : 0;
    auto label_result = load_textdata(labels_path, count);
    label_rd = std::move(label_result.first);
    label_strings = std::move(label_result.second);

    if (rd.data && label_rd.data) {
        out.data = rd;
        out.labels = label_rd;
        out.timeseries = timeseries;
    } else {
        free_realdata(rd);
        free_realdata(label_rd);
    }

    return {out, std::move(label_strings)};
}

/* Shuffle and split a loaded dataset. Shuffles in-place, splits into
 * train/test. Caller must free via free_dataset(). */
void split_dataset(Dataset* source, double train_percent,
                   Dataset* train, Dataset* test) {
    *train = make_empty_dataset();
    *test  = make_empty_dataset();
    assert(train_percent >= 0.00 && train_percent <= 1.00);

    if (!source->data.data || !source->labels.data) {
        return; // empty source
    }

    // Shuffle in-place (both data and labels together)
    shuffle_two(source->data.data, source->data.shape, source->data.dims,
                source->labels.data, source->labels.shape, source->labels.dims);

    int num_obs = source->data.shape[0];
    int train_len = (int)(train_percent * num_obs);
    int test_len  = num_obs - train_len;

    if (source->data.dims == 3) {
        int block_size = source->data.shape[1] * source->data.shape[2];
        int input_features = source->data.shape[1];
        build_split_dataset_3d(&source->data, &source->labels,
                               0, train_len, block_size, input_features, train);
        build_split_dataset_3d(&source->data, &source->labels,
                               train_len, test_len, block_size, input_features, test);
    } else {
        int cols = source->data.shape[1];
        build_split_dataset_2d(&source->data, &source->labels,
                               0, train_len, cols, train);
        build_split_dataset_2d(&source->data, &source->labels,
                               train_len, test_len, cols, test);
    }
}

/* Full dataset loading: load + shuffle + split into train/test.
 * Convenience wrapper around load_dataset_single + split_dataset.
 * label_strings filled with unique label names (same for train and test). */
void load_dataset(const char* data_path, const char* labels_path,
                  double train_percent, bool timeseries,
                  Dataset* train, Dataset* test,
                  std::vector<std::string>& label_strings) {
    auto result = load_dataset_single(data_path, labels_path, timeseries);
    split_dataset(&result.first, train_percent, train, test);
    free_dataset(&result.first);
    label_strings = std::move(result.second);
}
