#pragma once

typedef struct {
    double* data;
    double* labels;
    double* min_vals;
    double* max_vals;
    int rows;
    int cols;
} Dataset;

void load_dataset(const char* data_path, const char* labels_path,
                  double train_percent, Dataset* train, Dataset* test);
