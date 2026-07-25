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

/* Low-level loaders — file I/O and parsing encapsulated. Returns heap-allocated structs.
 * Caller must free via free_realdata() or free_textdata(). */
RealData load_realdata_2d(const char* path);       /* 2D data: rows x cols */
RealData load_realdata_3d(const char* path);       /* 3D data: obs x features x timesteps */
TextData load_textdata(const char* path, int count); /* text labels, count = number of observations */

/* High-level classification dataset loader. Loads both files, shuffles, and splits.
 * Caller must free train and test via free_classification_dataset(). */
void load_classification_dataset(const char* data_path, const char* labels_path,
                                  double train_percent, bool timeseries,
                                  ClassificationDataset* train, ClassificationDataset* test);

/* Low-level classification: load a single dataset (no split), then split separately. */
void load_classification_single(const char* data_path, const char* labels_path, bool timeseries,
                                 ClassificationDataset* out);

void split_classification(ClassificationDataset* source, double train_percent,
                          ClassificationDataset* train, ClassificationDataset* test);

/* Compute min/max values for RealData from its data buffer.
 * Uses shape/dims to determine iteration bounds.
 * Caller must allocate min_vals/max_vals arrays of correct size. */
void compute_realdata_minmax(RealData& rd);

/* Normalize RealData to [0,1] range using its min_vals/max_vals.
 * Values with zero range are set to NaN to match original behavior.
 * Modifies rd.data in-place; min_vals/max_vals unchanged. */
void normalize_realdata(RealData& rd);

/* Dataset cleanup */
void free_classification_dataset(ClassificationDataset* ds);
void free_realdata(RealData rd);
void free_textdata(TextData td);

/* Regression dataset loading */
void load_regression_dataset(const char* data_path, const char* target_path,
                              double train_percent, bool timeseries,
                              RegressionDataset* train, RegressionDataset* test);

void load_regression_dataset_single(const char* data_path, const char* target_path,
                                     bool timeseries,
                                     RegressionDataset* out);

void free_regression_dataset(RegressionDataset* ds);
