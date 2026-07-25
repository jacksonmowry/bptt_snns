#include "csv.h"
#include <assert.h>
#include <cfloat>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>

static int cmpstringp(const void* p1, const void* p2) {
    return strcmp(*(const char**)p1, *(const char**)p2);
}

// Build unique sorted label list from raw strings. Returns count.
static void build_label_mapping(char** raw_labels, int obs_count,
                                double* indices_out, char*** unique_labels_out,
                                int* unique_count_out) {
    char** unique = (char**)malloc(obs_count * sizeof(char*));
    int ucount    = 0;

    for (int i = 0; i < obs_count; i++) {
        bool found = false;
        for (int j = 0; j < ucount; j++) {
            if (strcmp(raw_labels[i], unique[j]) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            unique[ucount++] = raw_labels[i];
        }
    }

    qsort(unique, ucount, sizeof(char*), cmpstringp);

    for (int i = 0; i < obs_count; i++) {
        for (int j = 0; j < ucount; j++) {
            if (strcmp(raw_labels[i], unique[j]) == 0) {
                indices_out[i] = (double)j;
                break;
            }
        }
    }

    for (int i = 0; i < obs_count; i++) {
        bool is_unique = false;
        for (int j = 0; j < ucount; j++) {
            if (raw_labels[i] == unique[j]) {
                is_unique = true;
                break;
            }
        }
        if (!is_unique) {
            free(raw_labels[i]);
        }
    }
    free(raw_labels);

    *unique_labels_out = unique;
    *unique_count_out  = ucount;
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
    for (int j = 0; j < cols; j++) {
        rd.min_vals[j] = rd.data[j];
        rd.max_vals[j] = rd.data[j];
    }
    for (int i = 1; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double val = rd.data[(size_t)i * (size_t)cols + (size_t)j];
            if (val < rd.min_vals[j]) {
                rd.min_vals[j] = val;
            }
            if (val > rd.max_vals[j]) {
                rd.max_vals[j] = val;
            }
        }
    }

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
    for (int feature = 0; feature < input_features; feature++) {
        rd.min_vals[feature] = DBL_MAX;
        rd.max_vals[feature] = -DBL_MAX;
    }
    for (int obs = 0; obs < num_obs; obs++) {
        for (int feature = 0; feature < input_features; feature++) {
            for (int column = 0; column < timesteps; column++) {
                double val = rd.data[(obs * block_size) +
                                     (feature * timesteps) + column];
                if (val < rd.min_vals[feature]) {
                    rd.min_vals[feature] = val;
                }
                if (val > rd.max_vals[feature]) {
                    rd.max_vals[feature] = val;
                }
            }
        }
    }

    return rd;
}

// Parse labels from file.
// Returns TextData with data, label_strings, shape allocated.
// Caller must free: labels.data, labels.label_strings, labels.shape.
static TextData parse_textdata(FILE* f_labels, int count) {
    TextData td = {NULL, NULL, 0, 1, NULL};
    
    td.data     = (double*)malloc(count * sizeof(double));
    
    rewind(f_labels);
    char** raw_labels = read_label_strings(f_labels, count);
    
    build_label_mapping(raw_labels, count, td.data, &td.label_strings,
                        &td.label_strings_count);
    
    td.shape   = (int*)malloc(1 * sizeof(int));
    td.shape[0] = count;
    
    return td;
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
static void free_realdata(RealData rd) {
    free(rd.data);
    free(rd.min_vals);
    free(rd.max_vals);
    free(rd.shape);
}

// Free all members of a TextData.
static void free_textdata(TextData td) {
    free(td.data);
    for (int i = 0; i < td.label_strings_count; i++) {
        free(td.label_strings[i]);
    }
    free(td.label_strings);
    free(td.shape);
}

// Build a train/test ClassificationDataset from pre-shuffled RealData+TextData (2D, non-timeseries).
// Copies min_vals/max_vals from source and recomputes min/max on the split subsets.
// Each split owns its own copy of label_strings (no sharing). Caller must free via
// free_classification_dataset.
static void build_split_dataset_2d(const RealData* rd, const TextData* td,
                                    int start, int len, int cols,
                                    bool is_train, ClassificationDataset* out) {
    (void)is_train; // unused now that label_strings are always copied
    // Always copy data, min_vals, max_vals, and label_strings for each split.
    out->data.data     = (double*)malloc((size_t)len * (size_t)cols * sizeof(double));
    out->labels.data   = (double*)malloc(len * sizeof(double));
    out->data.min_vals = (double*)malloc(cols * sizeof(double));
    out->data.max_vals = (double*)malloc(cols * sizeof(double));
    if (!out->data.data || !out->labels.data || !out->data.min_vals || !out->data.max_vals) {
        free(out->data.data); free(out->labels.data);
        free(out->data.min_vals); free(out->data.max_vals);
        *out = {};
        return;
    }

    // Each split owns its own copy of label_strings (no sharing between train/test).
    out->labels.label_strings       = (char**)malloc(td->label_strings_count * sizeof(char*));
    out->labels.label_strings_count = td->label_strings_count;
    if (!out->labels.label_strings) {
        free(out->data.data); free(out->labels.data);
        free(out->data.min_vals); free(out->data.max_vals);
        *out = {};
        return;
    }
    for (int i = 0; i < td->label_strings_count; i++) {
        out->labels.label_strings[i] = strdup(td->label_strings[i]);
        if (!out->labels.label_strings[i]) {
            for (int j = 0; j < i; j++) {
                free(out->labels.label_strings[j]);
            }
            free(out->labels.label_strings);
            free(out->data.data); free(out->labels.data);
            free(out->data.min_vals); free(out->data.max_vals);
            *out = {};
            return;
        }
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
    memcpy(out->labels.data, td->data + start,
           len * sizeof(double));

    // Copy min/max from source
    memcpy(out->data.min_vals, rd->min_vals, cols * sizeof(double));
    memcpy(out->data.max_vals, rd->max_vals, cols * sizeof(double));

    // Recompute min/max on the split subset
    for (int col = 0; col < cols; col++) {
        out->data.min_vals[col] = out->data.data[col];
        out->data.max_vals[col] = out->data.data[col];
    }
    for (int row = 1; row < len; row++) {
        for (int col = 0; col < cols; col++) {
            double val = out->data.data[row * cols + col];
            if (val < out->data.min_vals[col]) {
                out->data.min_vals[col] = val;
            }
            if (val > out->data.max_vals[col]) {
                out->data.max_vals[col] = val;
            }
        }
    }
}

// Build a train/test ClassificationDataset from pre-shuffled RealData+TextData (3D, timeseries).
// Each split owns its own copy of label_strings (no sharing). Caller must free via
// free_classification_dataset.
static void build_split_dataset_3d(const RealData* rd, const TextData* td,
                                    int start, int len, int block_size,
                                    int input_features,
                                    bool /*is_train*/, ClassificationDataset* out) {
    out->data.data     = (double*)malloc(len * block_size * sizeof(double));
    out->labels.data   = (double*)malloc(len * sizeof(double));
    out->data.min_vals = (double*)malloc(input_features * sizeof(double));
    out->data.max_vals = (double*)malloc(input_features * sizeof(double));
    if (!out->data.data || !out->labels.data || !out->data.min_vals || !out->data.max_vals) {
        free(out->data.data); free(out->labels.data);
        free(out->data.min_vals); free(out->data.max_vals);
        *out = {};
        return;
    }

    // Each split owns its own copy of label_strings (no sharing between train/test).
    out->labels.label_strings       = (char**)malloc(td->label_strings_count * sizeof(char*));
    out->labels.label_strings_count = td->label_strings_count;
    if (!out->labels.label_strings) {
        free(out->data.data); free(out->labels.data);
        free(out->data.min_vals); free(out->data.max_vals);
        *out = {};
        return;
    }
    for (int i = 0; i < td->label_strings_count; i++) {
        out->labels.label_strings[i] = strdup(td->label_strings[i]);
        if (!out->labels.label_strings[i]) {
            for (int j = 0; j < i; j++) {
                free(out->labels.label_strings[j]);
            }
            free(out->labels.label_strings);
            free(out->data.data); free(out->labels.data);
            free(out->data.min_vals); free(out->data.max_vals);
            *out = {};
            return;
        }
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
    memcpy(out->labels.data, td->data + start,
           len * sizeof(double));
    memcpy(out->data.min_vals, rd->min_vals, input_features * sizeof(double));
    memcpy(out->data.max_vals, rd->max_vals, input_features * sizeof(double));
}

// Free a ClassificationDataset. All heap memory is freed.
// label_strings are also freed (each dataset owns its own copy).
void free_classification_dataset(ClassificationDataset* ds) {
    if (!ds) return;
    free_realdata(ds->data);
    if (ds->labels.label_strings) {
        for (int i = 0; i < ds->labels.label_strings_count; i++) {
            free(ds->labels.label_strings[i]);
        }
        free(ds->labels.label_strings);
    }
    free(ds->labels.data);
    free(ds->labels.shape);
    ds->data.data = NULL;
    ds->data.min_vals = NULL;
    ds->data.max_vals = NULL;
    ds->data.shape = NULL;
    ds->labels.data = NULL;
    ds->labels.label_strings = NULL;
    ds->labels.shape = NULL;
}

/* Load a single dataset from data+label files. Detects 2D vs 3D via the
 * timeseries flag. The dataset is NOT shuffled or split. Caller must free
 * via free_classification_dataset(). */
void load_dataset(const char* data_path, const char* labels_path, bool timeseries,
                  ClassificationDataset* out) {
    *out           = {{NULL, NULL, NULL, 0, NULL}, {NULL, NULL, 0, 0, NULL}, false};
    FILE* f_data   = fopen(data_path, "r");
    FILE* f_labels = fopen(labels_path, "r");
    if (f_data == NULL || f_labels == NULL) {
        if (f_data != NULL) fclose(f_data);
        if (f_labels != NULL) fclose(f_labels);
        return;
    }

    if (timeseries) {
        // 3D: timeseries data
        int num_obs, input_features, timesteps;
        compute_3d_dims(f_data, &num_obs, &input_features, &timesteps);
        fclose(f_data);
        fclose(f_labels);

        if (num_obs == 0) {
            *out = {};
            return;
        }

        // Reset file pointer for 3D parsing
        f_data = fopen(data_path, "r");
        f_labels = fopen(labels_path, "r");
        if (!f_data || !f_labels) {
            if (f_data) fclose(f_data);
            if (f_labels) fclose(f_labels);
            *out = {};
            return;
        }

        RealData rd = parse_realdata_3d(f_data, num_obs, input_features, timesteps);
        TextData td = parse_textdata(f_labels, num_obs);
        fclose(f_data);
        fclose(f_labels);

        if (!rd.data || !td.data || !rd.min_vals || !rd.max_vals) {
            free_realdata(rd);
            free_textdata(td);
            *out = {};
            return;
        }

        *out = {rd, td, true};
    } else {
        // 2D: non-timeseries data
        int cols;
        int rows = count_2d_lines(f_data, &cols);
        fclose(f_data);
        fclose(f_labels);

        if (rows == 0) {
            *out = {};
            return;
        }

        // Reset file pointers for 2D parsing
        f_data = fopen(data_path, "r");
        f_labels = fopen(labels_path, "r");
        if (!f_data || !f_labels) {
            if (f_data) fclose(f_data);
            if (f_labels) fclose(f_labels);
            *out = {};
            return;
        }

        RealData rd = parse_realdata_2d(f_data, rows, cols);
        TextData td = parse_textdata(f_labels, rows);
        fclose(f_data);
        fclose(f_labels);

        if (!rd.data || !td.data || !rd.min_vals || !rd.max_vals) {
            free_realdata(rd);
            free_textdata(td);
            *out = {};
            return;
        }

        *out = {rd, td, false};
    }
}

/* Shuffle and split a loaded dataset into train/test.
 * Shuffles the source dataset in-place, then splits into train (first train_len)
 * and test (remaining). Both train and test own independent label_strings copies.
 * Source dataset is left in a shuffled but otherwise unchanged state.
 * Caller must free train and test via free_classification_dataset(). */
void split_dataset(ClassificationDataset* source, double train_percent,
                   ClassificationDataset* train, ClassificationDataset* test) {
    *train = {};
    *test  = {};
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
        // 3D split
        int block_size = source->data.shape[1] * source->data.shape[2];
        int input_features = source->data.shape[1];
        build_split_dataset_3d(&source->data, &source->labels,
                               0, train_len, block_size, input_features, true, train);
        build_split_dataset_3d(&source->data, &source->labels,
                               train_len, test_len, block_size, input_features, false, test);
    } else {
        // 2D split
        int cols = source->data.shape[1];
        build_split_dataset_2d(&source->data, &source->labels,
                               0, train_len, cols, true, train);
        build_split_dataset_2d(&source->data, &source->labels,
                               train_len, test_len, cols, false, test);
    }
}
