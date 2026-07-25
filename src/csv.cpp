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
            assert(strlen(line) < sizeof(line) - 2);
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
    RealData rd = {NULL, NULL, NULL, 2, NULL};
    TextData td = {NULL, NULL, 0, 1, NULL};
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
    rd.shape    = (int*)malloc(2 * sizeof(int));
    rd.shape[0] = rows;

    int cols = 0;
    if (fgets(line, sizeof(line), f_data) != NULL) {
        for (int i = 0; line[i] != '\0' && line[i] != '\n'; i++) {
            if (line[i] == ',') {
                cols++;
            }
        }
        cols++;
    }
    rd.shape[1] = cols;

    rd.data     = (double*)malloc((size_t)rows * (size_t)cols * sizeof(double));
    td.data     = (double*)malloc(rows * sizeof(double));
    rd.min_vals = (double*)malloc(cols * sizeof(double));
    rd.max_vals = (double*)malloc(cols * sizeof(double));
    if (rd.data == NULL || td.data == NULL || rd.min_vals == NULL ||
        rd.max_vals == NULL) {
        free(rd.data);
        free(td.data);
        free(rd.min_vals);
        free(rd.max_vals);
        free(rd.shape);
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
                    rd.data[(size_t)i * (size_t)cols + (size_t)j] = atof(token);
                    token = strtok(NULL, ",");
                }
            }
        }
    }
    fclose(f_data);
    fclose(f_labels);

    // Build unique sorted label mapping
    build_label_mapping(raw_labels, rows, td.data, &td.label_strings,
                        &td.label_strings_count);

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

    // Shuffle (data + labels together)
    assert(rows >= 1);
    double* tmp = (double*)malloc(cols * sizeof(*tmp));
    for (size_t i = 1; i < (size_t)rows; i++) {
        size_t swap_idx = rand() % i;
        memcpy(tmp, rd.data + (i * cols), cols * sizeof(*rd.data));
        memcpy(rd.data + (i * cols), rd.data + (swap_idx * cols),
               cols * sizeof(*rd.data));
        memcpy(rd.data + (swap_idx * cols), tmp, cols * sizeof(*rd.data));
        double tmp_label    = td.data[i];
        td.data[i]          = td.data[swap_idx];
        td.data[swap_idx]   = tmp_label;
    }
    free(tmp);

    // Train/Test Split
    int train_len = (int)(train_percent * rows);
    int test_len  = rows - train_len;

    // Train DS
    train->data.data     = (double*)malloc((size_t)train_len * (size_t)cols *
                                      sizeof(*train->data.data));
    train->labels.data   = (double*)malloc(train_len * sizeof(*train->labels.data));
    train->data.min_vals = (double*)malloc(cols * sizeof(*train->data.min_vals));
    train->data.max_vals = (double*)malloc(cols * sizeof(*train->data.max_vals));
    if (!train->data.data || !train->labels.data || !train->data.min_vals ||
        !train->data.max_vals) {
        free(train->data.data);
        free(train->labels.data);
        free(train->data.min_vals);
        free(train->data.max_vals);
        free(rd.data);
        free(td.data);
        free(rd.min_vals);
        free(rd.max_vals);
        for (int i = 0; i < td.label_strings_count; i++) {
            free(td.label_strings[i]);
        }
        free(td.label_strings);
        *train = {};
        *test  = {};
        return;
    }
    train->labels.label_strings       = td.label_strings;
    train->labels.label_strings_count = td.label_strings_count;
    train->data.dims                  = 2;
    train->data.shape                 = (int*)malloc(2 * sizeof(int));
    train->data.shape[0]              = train_len;
    train->data.shape[1]              = cols;
    train->labels.dims                = 1;
    train->labels.shape               = (int*)malloc(1 * sizeof(int));
    train->labels.shape[0]            = train_len;
    train->timeseries                 = false;
    memcpy(train->data.min_vals, rd.min_vals, cols * sizeof(*train->data.min_vals));
    memcpy(train->data.max_vals, rd.max_vals, cols * sizeof(*train->data.max_vals));
    memcpy(train->data.data, rd.data, train_len * cols * sizeof(*train->data.data));
    memcpy(train->labels.data, td.data, train_len * sizeof(*train->labels.data));

    // Min/Max calcs for train only
    for (int col = 0; col < cols; col++) {
        train->data.min_vals[col] = train->data.data[col];
        train->data.max_vals[col] = train->data.data[col];
    }
    for (int row = 1; row < train_len; row++) {
        for (int col = 0; col < cols; col++) {
            double val = train->data.data[row * cols + col];
            if (val < train->data.min_vals[col]) {
                train->data.min_vals[col] = val;
            }
            if (val > train->data.max_vals[col]) {
                train->data.max_vals[col] = val;
            }
        }
    }

    // Test DS
    test->data.data =
        (double*)malloc((size_t)test_len * (size_t)cols * sizeof(*test->data.data));
    test->labels.data   = (double*)malloc(test_len * sizeof(*test->labels.data));
    test->data.min_vals = (double*)malloc(cols * sizeof(*test->data.min_vals));
    test->data.max_vals = (double*)malloc(cols * sizeof(*test->data.max_vals));
    if (!test->data.data || !test->labels.data || !test->data.min_vals || !test->data.max_vals) {
        free(test->data.data);
        free(test->labels.data);
        free(test->data.min_vals);
        free(test->data.max_vals);
        free(train->data.data);
        free(train->labels.data);
        free(train->data.min_vals);
        free(train->data.max_vals);
        free(rd.data);
        free(td.data);
        free(rd.min_vals);
        free(rd.max_vals);
        for (int i = 0; i < td.label_strings_count; i++) {
            free(td.label_strings[i]);
        }
        free(td.label_strings);
        *train = {};
        *test  = {};
        return;
    }
    test->labels.label_strings       = td.label_strings;
    test->labels.label_strings_count = td.label_strings_count;
    test->data.dims                  = 2;
    test->data.shape                 = (int*)malloc(2 * sizeof(int));
    test->data.shape[0]              = test_len;
    test->data.shape[1]              = cols;
    test->labels.dims                = 1;
    test->labels.shape               = (int*)malloc(1 * sizeof(int));
    test->labels.shape[0]            = test_len;
    test->timeseries                 = false;
    memcpy(test->data.min_vals, rd.min_vals, cols * sizeof(*test->data.min_vals));
    memcpy(test->data.max_vals, rd.max_vals, cols * sizeof(*test->data.max_vals));
    memcpy(test->data.data, rd.data + (train_len * cols),
           test_len * cols * sizeof(*test->data.data));
    memcpy(test->labels.data, td.data + train_len,
           test_len * sizeof(*test->labels.data));

    // Min/Max calcs for test only
    for (int col = 0; col < cols; col++) {
        test->data.min_vals[col] = test->data.data[col];
        test->data.max_vals[col] = test->data.data[col];
    }
    for (int row = 0; row < test_len; row++) {
        for (int col = 0; col < cols; col++) {
            double val = test->data.data[row * cols + col];
            if (val < test->data.min_vals[col]) {
                test->data.min_vals[col] = val;
            }
            if (val > test->data.max_vals[col]) {
                test->data.max_vals[col] = val;
            }
        }
    }

    free(rd.min_vals);
    free(rd.max_vals);
    free(rd.data);
    free(td.data);
    free(rd.shape);
}

void load_dataset_single(const char* data_path, const char* labels_path,
                         Dataset* out) {
    *out           = {{NULL, NULL, NULL, 2, NULL}, {NULL, NULL, 0, 1, NULL}, false};
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
    out->data.shape    = (int*)malloc(2 * sizeof(int));
    out->data.shape[0] = rows;

    int cols = 0;
    if (fgets(line, sizeof(line), f_data) != NULL) {
        for (int i = 0; line[i] != '\0' && line[i] != '\n'; i++) {
            if (line[i] == ',') {
                cols++;
            }
        }
        cols++;
    }
    out->data.shape[1] = cols;

    out->data.data   = (double*)malloc((size_t)rows * (size_t)cols * sizeof(double));
    out->labels.data = (double*)malloc(rows * sizeof(double));
    out->data.min_vals = (double*)malloc(cols * sizeof(double));
    out->data.max_vals = (double*)malloc(cols * sizeof(double));
    if (out->data.data == NULL || out->labels.data == NULL || out->data.min_vals == NULL ||
        out->data.max_vals == NULL) {
        free(out->data.data);
        free(out->labels.data);
        free(out->data.min_vals);
        free(out->data.max_vals);
        free(out->data.shape);
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
                    out->data.data[(size_t)i * (size_t)cols + (size_t)j] =
                        atof(token);
                    token = strtok(NULL, ",");
                }
            }
        }
    }
    fclose(f_data);
    fclose(f_labels);

    // Build unique sorted label mapping
    build_label_mapping(raw_labels, rows, out->labels.data, &out->labels.label_strings,
                        &out->labels.label_strings_count);

    // Allocate labels shape (1D)
    out->labels.dims    = 1;
    out->labels.shape   = (int*)malloc(1 * sizeof(int));
    out->labels.shape[0] = rows;

    // Min/max calcs
    for (int col = 0; col < cols; col++) {
        out->data.min_vals[col] = out->data.data[col];
        out->data.max_vals[col] = out->data.data[col];
    }
    for (int row = 1; row < out->data.shape[0]; row++) {
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

void load_dataset_2d(const char* data_path, const char* labels_path,
                     double train_percent, Dataset* train, Dataset* test) {
    assert(train_percent >= 0.0 && train_percent <= 1.0);
    RealData rd = {NULL, NULL, NULL, 3, NULL};
    TextData td = {NULL, NULL, 0, 1, NULL};
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

    rd.shape    = (int*)malloc(3 * sizeof(int));
    rd.shape[0] = num_obs;
    rd.shape[1] = input_features;
    rd.shape[2] = timesteps;

    int block_size = input_features * timesteps;
    int total_data = num_obs * block_size;
    rd.data        = (double*)malloc(total_data * sizeof(double));
    td.data        = (double*)malloc(num_obs * sizeof(double));
    rd.min_vals    = (double*)malloc(input_features * sizeof(double));
    rd.max_vals    = (double*)malloc(input_features * sizeof(double));

    if (rd.data == NULL || td.data == NULL || rd.min_vals == NULL ||
        rd.max_vals == NULL) {
        free(rd.data);
        free(td.data);
        free(rd.min_vals);
        free(rd.max_vals);
        free(rd.shape);
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
            rd.data[(obs_idx * block_size) + (line_cnt * timesteps) + c] =
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
    build_label_mapping(raw_labels, num_obs, td.data, &td.label_strings,
                        &td.label_strings_count);

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

    // Shuffle
    double* tmp_block = (double*)malloc(block_size * sizeof(double));
    for (int i = 1; i < num_obs; i++) {
        int swap_idx = rand() % i;
        memcpy(tmp_block, rd.data + (i * block_size),
               block_size * sizeof(double));
        memcpy(rd.data + (i * block_size), rd.data + (swap_idx * block_size),
               block_size * sizeof(double));
        memcpy(rd.data + (swap_idx * block_size), tmp_block,
               block_size * sizeof(double));
        double tmp_label    = td.data[i];
        td.data[i]          = td.data[swap_idx];
        td.data[swap_idx]   = tmp_label;
    }
    free(tmp_block);

    int train_len = (int)(train_percent * num_obs);
    int test_len  = num_obs - train_len;

    train->data.data     = (double*)malloc(train_len * block_size * sizeof(double));
    train->labels.data   = (double*)malloc(train_len * sizeof(double));
    train->data.min_vals = (double*)malloc(input_features * sizeof(double));
    train->data.max_vals = (double*)malloc(input_features * sizeof(double));
    if (!train->data.data || !train->labels.data || !train->data.min_vals ||
        !train->data.max_vals) {
        free(train->data.data);
        free(train->labels.data);
        free(train->data.min_vals);
        free(train->data.max_vals);
        free(rd.data);
        free(td.data);
        free(rd.min_vals);
        free(rd.max_vals);
        for (int i = 0; i < td.label_strings_count; i++) {
            free(td.label_strings[i]);
        }
        free(td.label_strings);
        *train = {};
        *test  = {};
        return;
    }
    train->labels.label_strings       = td.label_strings;
    train->labels.label_strings_count = td.label_strings_count;
    train->data.dims                  = 3;
    train->data.shape                 = (int*)malloc(3 * sizeof(int));
    train->data.shape[0]              = train_len;
    train->data.shape[1]              = input_features;
    train->data.shape[2]              = timesteps;
    train->labels.dims                = 1;
    train->labels.shape               = (int*)malloc(1 * sizeof(int));
    train->labels.shape[0]            = train_len;
    train->timeseries                 = true;
    memcpy(train->data.data, rd.data, train_len * block_size * sizeof(double));
    memcpy(train->labels.data, td.data, train_len * sizeof(double));
    memcpy(train->data.min_vals, rd.min_vals, input_features * sizeof(double));
    memcpy(train->data.max_vals, rd.max_vals, input_features * sizeof(double));

    test->data.data     = (double*)malloc(test_len * block_size * sizeof(double));
    test->labels.data   = (double*)malloc(test_len * sizeof(double));
    test->data.min_vals = (double*)malloc(input_features * sizeof(double));
    test->data.max_vals = (double*)malloc(input_features * sizeof(double));
    if (!test->data.data || !test->labels.data || !test->data.min_vals || !test->data.max_vals) {
        free(test->data.data);
        free(test->labels.data);
        free(test->data.min_vals);
        free(test->data.max_vals);
        free(train->data.data);
        free(train->labels.data);
        free(train->data.min_vals);
        free(train->data.max_vals);
        free(rd.data);
        free(td.data);
        free(rd.min_vals);
        free(rd.max_vals);
        for (int i = 0; i < td.label_strings_count; i++) {
            free(td.label_strings[i]);
        }
        free(td.label_strings);
        *train = {};
        *test  = {};
        return;
    }
    test->labels.label_strings       = td.label_strings;
    test->labels.label_strings_count = td.label_strings_count;
    test->data.dims                  = 3;
    test->data.shape                 = (int*)malloc(3 * sizeof(int));
    test->data.shape[0]              = test_len;
    test->data.shape[1]              = input_features;
    test->data.shape[2]              = timesteps;
    test->labels.dims                = 1;
    test->labels.shape               = (int*)malloc(1 * sizeof(int));
    test->labels.shape[0]            = test_len;
    test->timeseries                 = true;
    memcpy(test->data.data, rd.data + train_len * block_size,
           test_len * block_size * sizeof(double));
    memcpy(test->labels.data, td.data + train_len, test_len * sizeof(double));
    memcpy(test->data.min_vals, rd.min_vals, input_features * sizeof(double));
    memcpy(test->data.max_vals, rd.max_vals, input_features * sizeof(double));

    free(rd.data);
    free(td.data);
    free(rd.min_vals);
    free(rd.max_vals);
    free(rd.shape);
}

void load_dataset_2d_single(const char* data_path, const char* labels_path,
                            Dataset* out) {
    *out           = {{NULL, NULL, NULL, 3, NULL}, {NULL, NULL, 0, 1, NULL}, false};
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

    out->data.shape    = (int*)malloc(3 * sizeof(int));
    out->data.shape[0] = num_obs;
    out->data.shape[1] = input_features;
    out->data.shape[2] = timesteps;

    int block_size = input_features * timesteps;
    int total_data = num_obs * block_size;
    out->data.data      = (double*)malloc(total_data * sizeof(double));
    out->labels.data    = (double*)malloc(num_obs * sizeof(double));
    out->data.min_vals  = (double*)malloc(input_features * sizeof(double));
    out->data.max_vals  = (double*)malloc(input_features * sizeof(double));

    if (out->data.data == NULL || out->labels.data == NULL || out->data.min_vals == NULL ||
        out->data.max_vals == NULL) {
        free(out->data.data);
        free(out->labels.data);
        free(out->data.min_vals);
        free(out->data.max_vals);
        free(out->data.shape);
        fclose(f_data);
        fclose(f_labels);
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
            out->data.data[(obs_idx * block_size) + (line_cnt * timesteps) + c] =
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
    build_label_mapping(raw_labels, num_obs, out->labels.data, &out->labels.label_strings,
                        &out->labels.label_strings_count);

    // Allocate labels shape (1D)
    out->labels.dims    = 1;
    out->labels.shape   = (int*)malloc(1 * sizeof(int));
    out->labels.shape[0] = num_obs;

    // Min/max
    for (int feature = 0; feature < input_features; feature++) {
        out->data.min_vals[feature] = DBL_MAX;
        out->data.max_vals[feature] = -DBL_MAX;
    }
    for (int obs = 0; obs < num_obs; obs++) {
        for (int feature = 0; feature < input_features; feature++) {
            for (int column = 0; column < timesteps; column++) {
                double val = out->data.data[(obs * block_size) +
                                       (feature * timesteps) + column];
                if (val < out->data.min_vals[feature]) {
                    out->data.min_vals[feature] = val;
                }
                if (val > out->data.max_vals[feature]) {
                    out->data.max_vals[feature] = val;
                }
            }
        }
    }
}
