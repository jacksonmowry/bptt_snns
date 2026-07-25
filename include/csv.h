#pragma once

#include <stdbool.h>

/* Real-valued data: features, min/max normalization, shape */
typedef struct {
    double* data;    // feature values, heap-allocated
    double* min_vals;// per-dimension min, heap-allocated
    double* max_vals;// per-dimension max, heap-allocated
    int dims;        // number of dimensions (2 or 3)
    int* shape;      // heap-allocated array of size dims
} RealData;

/* Text/string labels: label values as double indices, label name strings */
typedef struct {
    double* data;              // label values as double indices, heap-allocated
    char** label_strings;      // unique sorted label strings, index = label value
    int label_strings_count;   // number of unique labels
    int dims;                  // always 1
    int* shape;                // heap-allocated array of size dims
} TextData;

/* Combined dataset: real data + text labels */
typedef struct {
    RealData data;
    TextData labels;
    bool timeseries;
} Dataset;

void load_dataset(const char* data_path, const char* labels_path,
                  double train_percent, Dataset* train, Dataset* test);

void load_dataset_single(const char* data_path, const char* labels_path,
                         Dataset* out);

void load_dataset_2d(const char* data_path, const char* labels_path,
                     double train_percent, Dataset* train, Dataset* test);

void load_dataset_2d_single(const char* data_path, const char* labels_path,
                            Dataset* out);
