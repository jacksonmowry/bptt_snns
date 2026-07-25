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

/* Combined dataset: real data + text labels (classification) */
typedef struct {
    RealData data;
    TextData labels;
    bool timeseries;
} ClassificationDataset;

/* Regression dataset: real input + real target */
typedef struct {
    RealData input;  // features
    RealData target; // targets
    bool timeseries;
} RegressionDataset;

void load_dataset(const char* data_path, const char* labels_path,
                  double train_percent, ClassificationDataset* train, ClassificationDataset* test);

void load_dataset_single(const char* data_path, const char* labels_path,
                         ClassificationDataset* out);

void load_dataset_2d(const char* data_path, const char* labels_path,
                     double train_percent, ClassificationDataset* train, ClassificationDataset* test);

void load_dataset_2d_single(const char* data_path, const char* labels_path,
                            ClassificationDataset* out);

/* Dataset cleanup */
void free_classification_dataset(ClassificationDataset* ds);

/* Regression dataset loading (future) */
// void load_regression_dataset(const char* data_path, const char* target_path,
//                              double train_percent, RegressionDataset* train, RegressionDataset* test);
// void load_regression_dataset_single(const char* data_path, const char* target_path,
//                                     RegressionDataset* out);
