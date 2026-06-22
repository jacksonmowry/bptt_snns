#include "csv.h"
#include <assert.h>
#include <cstddef>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void load_dataset(const char* data_path, const char* labels_path,
                  double train_percent, Dataset* train, Dataset* test) {
    assert(train_percent >= 0.00 && train_percent <= 1.00);
    Dataset ds     = {NULL, NULL, NULL, NULL, 0, 0};
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
    int rows = 0;
    while (fgets(line, sizeof(line), f_labels) != NULL) {
        rows++;
    }
    ds.rows = rows;

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
        .rows     = train_len,
        .cols     = cols};
    memcpy(train->min_vals, ds.min_vals, cols * sizeof(*train->min_vals));
    memcpy(train->max_vals, ds.max_vals, cols * sizeof(*train->max_vals));
    memcpy(train->data, ds.data, train_len * cols * sizeof(*train->data));
    memcpy(train->labels, ds.labels, train_len * sizeof(*train->labels));

    // Test DS
    *test = (Dataset){
        .data     = (double*)malloc(test_len * cols * sizeof(*test->data)),
        .labels   = (double*)malloc(test_len * sizeof(*test->labels)),
        .min_vals = (double*)malloc(cols * sizeof(*test->min_vals)),
        .max_vals = (double*)malloc(cols * sizeof(*test->max_vals)),
        .rows     = test_len,
        .cols     = cols};
    memcpy(test->min_vals, ds.min_vals, cols * sizeof(*test->min_vals));
    memcpy(test->max_vals, ds.max_vals, cols * sizeof(*test->max_vals));
    memcpy(test->data, ds.data, test_len * cols * sizeof(*test->data));
    memcpy(test->labels, ds.labels, test_len * sizeof(*test->labels));

    free(ds.min_vals);
    free(ds.max_vals);
    free(ds.data);
    free(ds.labels);

    return;
}
