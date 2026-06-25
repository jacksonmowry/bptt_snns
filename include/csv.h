#pragma once

#include <stdbool.h>

typedef struct {
    float* data;
    float* processed_data;
    float* labels;
    float* min_vals;
    float* max_vals;
    int observations;
    int cols;
    int rows_per_observation;
    bool timeseries;
} Dataset;

void load_dataset(const char* data_path, const char* labels_path,
                  float train_percent, Dataset* train, Dataset* test);

void load_dataset_2d(const char* data_path, const char* labels_path,
                     float train_percent, Dataset* train, Dataset* test);
