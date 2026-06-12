#pragma once

typedef struct {
    double* data;
    double* labels;
    double* min_vals;
    double* max_vals;
    int rows;
    int cols;
} Dataset;

Dataset load_dataset(const char* data_path, const char* labels_path);
