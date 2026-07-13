#pragma once

#include <stdbool.h>

typedef struct {
    double* data;
    double* labels;
    double* min_vals;
    double* max_vals;
    char** label_strings;       // unique sorted label strings, index = label value
    int label_strings_count;   // number of unique labels
    int observations;
    int cols;
    int rows_per_observation;
    bool timeseries;
} Dataset;

void load_dataset(const char* data_path, const char* labels_path,
                  double train_percent, Dataset* train, Dataset* test);

void load_dataset_single(const char* data_path, const char* labels_path,
                         Dataset* out);

void load_dataset_2d(const char* data_path, const char* labels_path,
                     double train_percent, Dataset* train, Dataset* test);
