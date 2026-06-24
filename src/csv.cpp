#include "csv.h"
#include <assert.h>
#include <cfloat>
#include <cstddef>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void load_dataset(const char* data_path, const char* labels_path,
                  double train_percent, Dataset* train, Dataset* test) {
    assert(train_percent >= 0.00 && train_percent <= 1.00);
    Dataset ds     = {NULL, NULL, NULL, NULL, 0, 0, 0, 0};
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
    ds.observations = rows;

    int cols = 0;
    if (fgets(line, sizeof(line), f_data) != NULL) {
        for (int i = 0; line[i] != '\0' && line[i] != '\n'; i++) {
            if (line[i] == ',') {
                cols++;
            }
        }
        cols++;
    }
    ds.cols = cols;

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

    rewind(f_labels);
    rewind(f_data);
    for (int i = 0; i < rows; i++) {
        if (fgets(line, sizeof(line), f_labels) != NULL) {
            ds.labels[i] = atof(line);
        }
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

    // Shuffle
    assert(rows >= 1);
    double* tmp = (double*)malloc(cols * sizeof(*tmp));
    for (size_t i = 1; i < (size_t)rows; i++) {
        size_t swap_idx = rand() % i;

        // Current -> tmp
        // swap -> Current
        // tmp -> swap
        memcpy(tmp, ds.data + (i * cols), cols * sizeof(*ds.data));
        memcpy(ds.data + (i * cols), ds.data + (swap_idx * cols),
               cols * sizeof(*ds.data));
        memcpy(ds.data + (swap_idx * cols), tmp, cols * sizeof(*ds.data));

        double tmp_label    = ds.labels[i];
        ds.labels[i]        = ds.labels[swap_idx];
        ds.labels[swap_idx] = tmp_label;
    }

    // Train/Test Split
    int train_len = train_percent * rows;
    int test_len  = rows - train_len;

    // Train DS
    *train = (Dataset){
        .data     = (double*)malloc(train_len * cols * sizeof(*train->data)),
        .labels   = (double*)malloc(train_len * sizeof(*train->labels)),
        .min_vals = (double*)malloc(cols * sizeof(*train->min_vals)),
        .max_vals = (double*)malloc(cols * sizeof(*train->max_vals)),
        .observations         = train_len,
        .cols                 = cols,
        .rows_per_observation = -1,
        .timeseries           = false};
    memcpy(train->min_vals, ds.min_vals, cols * sizeof(*train->min_vals));
    memcpy(train->max_vals, ds.max_vals, cols * sizeof(*train->max_vals));
    memcpy(train->data, ds.data, train_len * cols * sizeof(*train->data));
    memcpy(train->labels, ds.labels, train_len * sizeof(*train->labels));

    // Test DS
    *test = (Dataset){
        .data         = (double*)malloc(test_len * cols * sizeof(*test->data)),
        .labels       = (double*)malloc(test_len * sizeof(*test->labels)),
        .min_vals     = (double*)malloc(cols * sizeof(*test->min_vals)),
        .max_vals     = (double*)malloc(cols * sizeof(*test->max_vals)),
        .observations = test_len,
        .cols         = cols,
        .rows_per_observation = -1,
        .timeseries           = false};
    memcpy(test->min_vals, ds.min_vals, cols * sizeof(*test->min_vals));
    memcpy(test->max_vals, ds.max_vals, cols * sizeof(*test->max_vals));
    memcpy(test->data, ds.data + (train_len * cols),
           test_len * cols * sizeof(*test->data));
    memcpy(test->labels, ds.labels + (train_len),
           test_len * sizeof(*test->labels));

    free(ds.min_vals);
    free(ds.max_vals);
    free(ds.data);
    free(ds.labels);

    return;
}

void load_dataset_2d(const char* data_path, const char* labels_path,
                     double train_percent, Dataset* train, Dataset* test) {
    assert(train_percent >= 0.0 && train_percent <= 1.0);
    Dataset ds     = {NULL, NULL, NULL, NULL, 0, 0, 0, 0};
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

    int N = non_empty / num_obs;
    rewind(f_data);

    int D = 0;
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
                    D++;
                }
            }
            D++;
            break;
        }
    }

    ds.observations         = num_obs;
    ds.rows_per_observation = N;
    ds.cols                 = D;

    int block_size = N * D;
    int total_data = num_obs * block_size;
    ds.data        = (double*)malloc(total_data * sizeof(double));
    ds.labels      = (double*)malloc(num_obs * sizeof(double));
    ds.min_vals    = (double*)malloc(N * sizeof(double));
    ds.max_vals    = (double*)malloc(N * sizeof(double));

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
        for (int c = 0; c < D && token != NULL; c++) {
            ds.data[(obs_idx * block_size) + (line_cnt * D) + c] = atof(token);
            token = strtok(NULL, ",");
        }
        line_cnt++;
    }

    int lbl_cnt = 0;
    while (fgets(line, sizeof(line), f_labels) != NULL) {
        int is_blank = 1;
        for (int k = 0; line[k]; k++) {
            if (line[k] != '\n' && line[k] != '\r' && line[k] != ' ') {
                is_blank = 0;
                break;
            }
        }
        if (!is_blank) {
            ds.labels[lbl_cnt++] = atof(line);
        }
    }

    fclose(f_data);
    fclose(f_labels);

    for (int feature = 0; feature < N; feature++) {
        ds.min_vals[feature] = DBL_MAX;
        ds.max_vals[feature] = -DBL_MAX;
    }
    for (int obs = 0; obs < num_obs; obs++) {
        for (int feature = 0; feature < N; feature++) {
            for (int column = 0; column < D; column++) {
                double val =
                    ds.data[(obs * block_size) + (feature * D) + column];

                if (val < ds.min_vals[feature]) {
                    ds.min_vals[feature] = val;
                }
                if (val > ds.max_vals[feature]) {
                    ds.max_vals[feature] = val;
                }
            }
        }
    }

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

    *train = (Dataset){
        .data     = (double*)malloc(train_len * block_size * sizeof(double)),
        .labels   = (double*)malloc(train_len * sizeof(double)),
        .min_vals = (double*)malloc(D * sizeof(double)),
        .max_vals = (double*)malloc(D * sizeof(double)),
        .observations         = train_len,
        .cols                 = D,
        .rows_per_observation = N,
        .timeseries           = true};
    memcpy(train->data, ds.data, train_len * block_size * sizeof(double));
    memcpy(train->labels, ds.labels, train_len * sizeof(double));
    memcpy(train->min_vals, ds.min_vals, N * sizeof(double));
    memcpy(train->max_vals, ds.max_vals, N * sizeof(double));

    *test = (Dataset){
        .data         = (double*)malloc(test_len * block_size * sizeof(double)),
        .labels       = (double*)malloc(test_len * sizeof(double)),
        .min_vals     = (double*)malloc(N * sizeof(double)),
        .max_vals     = (double*)malloc(N * sizeof(double)),
        .observations = test_len,
        .cols         = D,
        .rows_per_observation = N,
        .timeseries           = true};
    memcpy(test->data, ds.data + train_len * block_size,
           test_len * block_size * sizeof(double));
    memcpy(test->labels, ds.labels + train_len, test_len * sizeof(double));
    memcpy(test->min_vals, ds.min_vals, N * sizeof(double));
    memcpy(test->max_vals, ds.max_vals, N * sizeof(double));

    free(ds.data);
    free(ds.labels);
    free(ds.min_vals);
    free(ds.max_vals);
}
