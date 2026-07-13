#include "csv.h"
#include <assert.h>
#include <cfloat>
#include <cstddef>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cmpstringp(const void* p1, const void* p2) {
    return strcmp(*(const char**)p1, *(const char**)p2);
}

// Build unique sorted label list from raw strings. Returns count.
static void build_label_mapping(char** raw_labels, int obs_count,
                                double* indices_out, char*** unique_labels_out,
                                int* unique_count_out) {
    // Collect unique labels (preserve first occurrence order, we sort later)
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

    // Sort with natural sort
    qsort(unique, ucount, sizeof(char*), cmpstringp);

    // Build index for each observation
    for (int i = 0; i < obs_count; i++) {
        for (int j = 0; j < ucount; j++) {
            if (strcmp(raw_labels[i], unique[j]) == 0) {
                indices_out[i] = (double)j;
                break;
            }
        }
    }

    // Free raw labels array (strings are now owned by unique)
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
            // strip newline
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

void load_dataset(const char* data_path, const char* labels_path,
                  double train_percent, Dataset* train, Dataset* test) {
    assert(train_percent >= 0.00 && train_percent <= 1.00);
    Dataset ds = {
        NULL, NULL, NULL, NULL, NULL, 0, 2, NULL, 0,
    };
    FILE* f_data   = fopen(data_path, "r");
    FILE* f_labels = fopen(labels_path, "r");
    if (f_data == NULL || f_labels == NULL) {
        if (f_data != NULL) {
            fclose(f_data);
        }
        if (f_labels != NULL) {
            fclose(f_labels);
        }
        *train = {};
        *test  = {};
        return;
    }

    char line[4096 * 16];
    int rows = 0;
    while (fgets(line, sizeof(line), f_labels) != NULL) {
        rows++;
    }
    ds.shape    = (int*)malloc(2 * sizeof(int));
    ds.shape[0] = rows;

    int cols = 0;
    if (fgets(line, sizeof(line), f_data) != NULL) {
        for (int i = 0; line[i] != '\0' && line[i] != '\n'; i++) {
            if (line[i] == ',') {
                cols++;
            }
        }
        cols++;
    }
    ds.shape[1] = cols;

    ds.data     = (double*)malloc(rows * cols * sizeof(double));
    ds.labels   = (double*)malloc(rows * sizeof(double));
    ds.min_vals = (double*)malloc(cols * sizeof(double));
    ds.max_vals = (double*)malloc(cols * sizeof(double));
    if (ds.data == NULL || ds.labels == NULL || ds.min_vals == NULL ||
        ds.max_vals == NULL) {
        free(ds.data);
        free(ds.labels);
        free(ds.min_vals);
        free(ds.max_vals);
        fclose(f_data);
        fclose(f_labels);
        *train = {};
        *test  = {};
        return;
    }

    // Read labels as strings
    rewind(f_labels);
    char** raw_labels = read_label_strings(f_labels, rows);

    // Read data
    rewind(f_data);
    for (int i = 0; i < rows; i++) {
        if (fgets(line, sizeof(line), f_data) != NULL) {
            char* token = strtok(line, ",");
            for (int j = 0; j < cols; j++) {
                if (token != NULL) {
                    ds.data[i * cols + j] = atof(token);
                    token                 = strtok(NULL, ",");
                }
            }
        }
    }
    fclose(f_data);
    fclose(f_labels);

    // Build unique sorted label mapping
    build_label_mapping(raw_labels, rows, ds.labels, &ds.label_strings,
                        &ds.label_strings_count);

    // Min/max calcs
    for (int j = 0; j < cols; j++) {
        ds.min_vals[j] = ds.data[j];
        ds.max_vals[j] = ds.data[j];
    }
    for (int i = 1; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double val = ds.data[i * cols + j];
            if (val < ds.min_vals[j]) {
                ds.min_vals[j] = val;
            }
            if (val > ds.max_vals[j]) {
                ds.max_vals[j] = val;
            }
        }
    }

    // Shuffle (data + labels together)
    assert(rows >= 1);
    double* tmp = (double*)malloc(cols * sizeof(*tmp));
    for (size_t i = 1; i < (size_t)rows; i++) {
        size_t swap_idx = rand() % i;
        memcpy(tmp, ds.data + (i * cols), cols * sizeof(*ds.data));
        memcpy(ds.data + (i * cols), ds.data + (swap_idx * cols),
               cols * sizeof(*ds.data));
        memcpy(ds.data + (swap_idx * cols), tmp, cols * sizeof(*ds.data));
        double tmp_label    = ds.labels[i];
        ds.labels[i]        = ds.labels[swap_idx];
        ds.labels[swap_idx] = tmp_label;
    }
    free(tmp);

    // Train/Test Split
    int train_len = (int)(train_percent * rows);
    int test_len  = rows - train_len;

    // Train DS
    train->data     = (double*)malloc(train_len * cols * sizeof(*train->data));
    train->labels   = (double*)malloc(train_len * sizeof(*train->labels));
    train->min_vals = (double*)malloc(cols * sizeof(*train->min_vals));
    train->max_vals = (double*)malloc(cols * sizeof(*train->max_vals));
    if (!train->data || !train->labels || !train->min_vals ||
        !train->max_vals) {
        free(train->data);
        free(train->labels);
        free(train->min_vals);
        free(train->max_vals);
        free(ds.data);
        free(ds.labels);
        free(ds.min_vals);
        free(ds.max_vals);
        for (int i = 0; i < ds.label_strings_count; i++) {
            free(ds.label_strings[i]);
        }
        free(ds.label_strings);
        *train = {};
        *test  = {};
        return;
    }
    train->label_strings       = ds.label_strings;
    train->label_strings_count = ds.label_strings_count;
    train->dims                = 2;
    train->shape               = (int*)malloc(2 * sizeof(int));
    train->shape[0]            = train_len;
    train->shape[1]            = cols;
    train->timeseries          = false;
    memcpy(train->min_vals, ds.min_vals, cols * sizeof(*train->min_vals));
    memcpy(train->max_vals, ds.max_vals, cols * sizeof(*train->max_vals));
    memcpy(train->data, ds.data, train_len * cols * sizeof(*train->data));
    memcpy(train->labels, ds.labels, train_len * sizeof(*train->labels));

    // Min/Max calcs for train only
    for (int col = 0; col < cols; col++) {
        train->min_vals[col] = train->data[col];
        train->max_vals[col] = train->data[col];
    }
    for (int row = 1; row < train_len; row++) {
        for (int col = 0; col < cols; col++) {
            double val = train->data[row * cols + col];
            if (val < train->min_vals[col]) {
                train->min_vals[col] = val;
            }
            if (val > train->max_vals[col]) {
                train->max_vals[col] = val;
            }
        }
    }

    // Test DS
    test->data     = (double*)malloc(test_len * cols * sizeof(*test->data));
    test->labels   = (double*)malloc(test_len * sizeof(*test->labels));
    test->min_vals = (double*)malloc(cols * sizeof(*test->min_vals));
    test->max_vals = (double*)malloc(cols * sizeof(*test->max_vals));
    if (!test->data || !test->labels || !test->min_vals || !test->max_vals) {
        free(test->data);
        free(test->labels);
        free(test->min_vals);
        free(test->max_vals);
        free(train->data);
        free(train->labels);
        free(train->min_vals);
        free(train->max_vals);
        free(ds.data);
        free(ds.labels);
        free(ds.min_vals);
        free(ds.max_vals);
        for (int i = 0; i < ds.label_strings_count; i++) {
            free(ds.label_strings[i]);
        }
        free(ds.label_strings);
        *train = {};
        *test  = {};
        return;
    }
    test->label_strings       = ds.label_strings;
    test->label_strings_count = ds.label_strings_count;
    test->dims                = 2;
    test->shape               = (int*)malloc(2 * sizeof(int));
    test->shape[0]            = test_len;
    test->shape[1]            = cols;
    test->timeseries          = false;
    memcpy(test->min_vals, ds.min_vals, cols * sizeof(*test->min_vals));
    memcpy(test->max_vals, ds.max_vals, cols * sizeof(*test->max_vals));
    memcpy(test->data, ds.data + (train_len * cols),
           test_len * cols * sizeof(*test->data));
    memcpy(test->labels, ds.labels + train_len,
           test_len * sizeof(*test->labels));

    // Min/Max calcs for test only
    for (int col = 0; col < cols; col++) {
        test->min_vals[col] = test->data[col];
        test->max_vals[col] = test->data[col];
    }
    for (int row = 0; row < test_len; row++) {
        for (int col = 0; col < cols; col++) {
            double val = test->data[row * cols + col];
            if (val < test->min_vals[col]) {
                test->min_vals[col] = val;
            }
            if (val > test->max_vals[col]) {
                test->max_vals[col] = val;
            }
        }
    }

    free(ds.min_vals);
    free(ds.max_vals);
    free(ds.data);
    free(ds.labels);
    free(ds.shape);
}

void load_dataset_single(const char* data_path, const char* labels_path,
                         Dataset* out) {
    *out           = {NULL, NULL, NULL, NULL, NULL, 0, 2, NULL, false};
    FILE* f_data   = fopen(data_path, "r");
    FILE* f_labels = fopen(labels_path, "r");
    if (f_data == NULL || f_labels == NULL) {
        if (f_data != NULL) {
            fclose(f_data);
        }
        if (f_labels != NULL) {
            fclose(f_labels);
        }
        return;
    }

    char line[4096 * 16];
    int rows = 0;
    while (fgets(line, sizeof(line), f_labels) != NULL) {
        rows++;
    }
    out->shape    = (int*)malloc(2 * sizeof(int));
    out->shape[0] = rows;

    int cols = 0;
    if (fgets(line, sizeof(line), f_data) != NULL) {
        for (int i = 0; line[i] != '\0' && line[i] != '\n'; i++) {
            if (line[i] == ',') {
                cols++;
            }
        }
        cols++;
    }
    out->shape[1] = cols;

    out->data     = (double*)malloc(rows * cols * sizeof(double));
    out->labels   = (double*)malloc(rows * sizeof(double));
    out->min_vals = (double*)malloc(cols * sizeof(double));
    out->max_vals = (double*)malloc(cols * sizeof(double));
    if (out->data == NULL || out->labels == NULL || out->min_vals == NULL ||
        out->max_vals == NULL) {
        free(out->data);
        free(out->labels);
        free(out->min_vals);
        free(out->max_vals);
        free(out->shape);
        fclose(f_data);
        fclose(f_labels);
        return;
    }

    // Read labels as strings
    rewind(f_labels);
    char** raw_labels = read_label_strings(f_labels, rows);

    // Read data
    rewind(f_data);
    for (int i = 0; i < rows; i++) {
        if (fgets(line, sizeof(line), f_data) != NULL) {
            char* token = strtok(line, ",");
            for (int j = 0; j < cols; j++) {
                if (token != NULL) {
                    out->data[i * cols + j] = atof(token);
                    token                   = strtok(NULL, ",");
                }
            }
        }
    }
    fclose(f_data);
    fclose(f_labels);

    // Build unique sorted label mapping
    build_label_mapping(raw_labels, rows, out->labels, &out->label_strings,
                        &out->label_strings_count);

    // Min/max calcs
    for (int col = 0; col < cols; col++) {
        out->min_vals[col] = out->data[col];
        out->max_vals[col] = out->data[col];
    }
    for (int row = 1; row < out->shape[0]; row++) {
        for (int col = 0; col < cols; col++) {
            double val = out->data[row * cols + col];
            if (val < out->min_vals[col]) {
                out->min_vals[col] = val;
            }
            if (val > out->max_vals[col]) {
                out->max_vals[col] = val;
            }
        }
    }
}

void load_dataset_2d(const char* data_path, const char* labels_path,
                     double train_percent, Dataset* train, Dataset* test) {
    assert(train_percent >= 0.0 && train_percent <= 1.0);
    Dataset ds     = {NULL, NULL, NULL, NULL, NULL, 0, 3, NULL, false};
    FILE* f_data   = fopen(data_path, "r");
    FILE* f_labels = fopen(labels_path, "r");

    if (f_data == NULL || f_labels == NULL) {
        if (f_data != NULL) {
            fclose(f_data);
        }
        if (f_labels != NULL) {
            fclose(f_labels);
        }
        *train = {};
        *test  = {};
        return;
    }

    char line[4096];
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

    if (num_obs == 0) {
        fclose(f_data);
        fclose(f_labels);
        *train = {};
        *test  = {};
        return;
    }

    int input_features = non_empty / num_obs;
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

    ds.shape    = (int*)malloc(3 * sizeof(int));
    ds.shape[0] = num_obs;
    ds.shape[1] = input_features;
    ds.shape[2] = timesteps;

    int block_size = input_features * timesteps;
    int total_data = num_obs * block_size;
    ds.data        = (double*)malloc(total_data * sizeof(double));
    ds.labels      = (double*)malloc(num_obs * sizeof(double));
    ds.min_vals    = (double*)malloc(input_features * sizeof(double));
    ds.max_vals    = (double*)malloc(input_features * sizeof(double));

    if (ds.data == NULL || ds.labels == NULL || ds.min_vals == NULL ||
        ds.max_vals == NULL) {
        free(ds.data);
        free(ds.labels);
        free(ds.min_vals);
        free(ds.max_vals);
        fclose(f_data);
        fclose(f_labels);
        *train = {};
        *test  = {};
        return;
    }

    rewind(f_data);
    rewind(f_labels);

    // Read data
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
            ds.data[(obs_idx * block_size) + (line_cnt * timesteps) + c] =
                atof(token);
            token = strtok(NULL, ",");
        }
        line_cnt++;
    }

    // Read labels as strings
    rewind(f_labels);
    char** raw_labels = read_label_strings(f_labels, num_obs);

    fclose(f_data);
    fclose(f_labels);

    // Build unique sorted label mapping
    build_label_mapping(raw_labels, num_obs, ds.labels, &ds.label_strings,
                        &ds.label_strings_count);

    // Min/max
    for (int feature = 0; feature < input_features; feature++) {
        ds.min_vals[feature] = DBL_MAX;
        ds.max_vals[feature] = -DBL_MAX;
    }
    for (int obs = 0; obs < num_obs; obs++) {
        for (int feature = 0; feature < input_features; feature++) {
            for (int column = 0; column < timesteps; column++) {
                double val = ds.data[(obs * block_size) +
                                     (feature * timesteps) + column];
                if (val < ds.min_vals[feature]) {
                    ds.min_vals[feature] = val;
                }
                if (val > ds.max_vals[feature]) {
                    ds.max_vals[feature] = val;
                }
            }
        }
    }

    // Shuffle
    double* tmp_block = (double*)malloc(block_size * sizeof(double));
    for (int i = 1; i < num_obs; i++) {
        int swap_idx = rand() % i;
        memcpy(tmp_block, ds.data + (i * block_size),
               block_size * sizeof(double));
        memcpy(ds.data + (i * block_size), ds.data + (swap_idx * block_size),
               block_size * sizeof(double));
        memcpy(ds.data + (swap_idx * block_size), tmp_block,
               block_size * sizeof(double));
        double tmp_label    = ds.labels[i];
        ds.labels[i]        = ds.labels[swap_idx];
        ds.labels[swap_idx] = tmp_label;
    }
    free(tmp_block);

    int train_len = (int)(train_percent * num_obs);
    int test_len  = num_obs - train_len;

    train->data     = (double*)malloc(train_len * block_size * sizeof(double));
    train->labels   = (double*)malloc(train_len * sizeof(double));
    train->min_vals = (double*)malloc(input_features * sizeof(double));
    train->max_vals = (double*)malloc(input_features * sizeof(double));
    if (!train->data || !train->labels || !train->min_vals ||
        !train->max_vals) {
        free(train->data);
        free(train->labels);
        free(train->min_vals);
        free(train->max_vals);
        free(ds.data);
        free(ds.labels);
        free(ds.min_vals);
        free(ds.max_vals);
        for (int i = 0; i < ds.label_strings_count; i++) {
            free(ds.label_strings[i]);
        }
        free(ds.label_strings);
        *train = {};
        *test  = {};
        return;
    }
    train->label_strings       = ds.label_strings;
    train->label_strings_count = ds.label_strings_count;
    train->dims                = 3;
    train->shape               = (int*)malloc(3 * sizeof(int));
    train->shape[0]            = train_len;
    train->shape[1]            = input_features;
    train->shape[2]            = timesteps;
    train->timeseries          = true;
    memcpy(train->data, ds.data, train_len * block_size * sizeof(double));
    memcpy(train->labels, ds.labels, train_len * sizeof(double));
    memcpy(train->min_vals, ds.min_vals, input_features * sizeof(double));
    memcpy(train->max_vals, ds.max_vals, input_features * sizeof(double));

    test->data     = (double*)malloc(test_len * block_size * sizeof(double));
    test->labels   = (double*)malloc(test_len * sizeof(double));
    test->min_vals = (double*)malloc(input_features * sizeof(double));
    test->max_vals = (double*)malloc(input_features * sizeof(double));
    if (!test->data || !test->labels || !test->min_vals || !test->max_vals) {
        free(test->data);
        free(test->labels);
        free(test->min_vals);
        free(test->max_vals);
        free(train->data);
        free(train->labels);
        free(train->min_vals);
        free(train->max_vals);
        free(ds.data);
        free(ds.labels);
        free(ds.min_vals);
        free(ds.max_vals);
        for (int i = 0; i < ds.label_strings_count; i++) {
            free(ds.label_strings[i]);
        }
        free(ds.label_strings);
        *train = {};
        *test  = {};
        return;
    }
    test->label_strings       = ds.label_strings;
    test->label_strings_count = ds.label_strings_count;
    test->dims                = 3;
    test->shape               = (int*)malloc(3 * sizeof(int));
    test->shape[0]            = test_len;
    test->shape[1]            = input_features;
    test->shape[2]            = timesteps;
    test->timeseries          = true;
    memcpy(test->data, ds.data + train_len * block_size,
           test_len * block_size * sizeof(double));
    memcpy(test->labels, ds.labels + train_len, test_len * sizeof(double));
    memcpy(test->min_vals, ds.min_vals, input_features * sizeof(double));
    memcpy(test->max_vals, ds.max_vals, input_features * sizeof(double));

    free(ds.data);
    free(ds.labels);
    free(ds.min_vals);
    free(ds.max_vals);
    free(ds.shape);
}
