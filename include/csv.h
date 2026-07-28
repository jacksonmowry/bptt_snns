#pragma once

#include <stdbool.h>
#include <vector>
#include <string>
#include <utility>

/* Real-valued data: features, min/max normalization, shape */
typedef struct {
    double* data;    // feature values, heap-allocated
    double* min_vals;// per-dimension min, heap-allocated
    double* max_vals;// per-dimension max, heap-allocated
    int dims;        // number of dimensions (2 or 3)
    int* shape;      // heap-allocated array of size dims
} RealData;

/* Combined dataset: real data + real label indices (classification) */
typedef struct {
    RealData data;      // input features
    RealData labels;    // label indices (dims=1, shape=[count])
    bool timeseries;
} ClassificationDataset;

/* Regression dataset: real input + real target */
typedef struct {
    RealData input;  // features
    RealData target; // targets
    bool timeseries;
} RegressionDataset;

/* Low-level loaders — file I/O and parsing encapsulated. Returns heap-allocated structs.
 * Caller must free via free_realdata(). */
RealData load_realdata_2d(const char* path);       /* 2D data: rows x cols */
RealData load_realdata_3d(const char* path);       /* 3D data: obs x features x timesteps */

/* Parse labels from file. Returns {label indices as RealData (dims=1), vector of label strings}.
 * count = number of observations (rows) in the file.
 * Caller must free the RealData's data/shape via free_realdata(). */
std::pair<RealData, std::vector<std::string>> load_textdata(const char* path, int count);

/* High-level classification dataset loader. Loads both files, shuffles, and splits.
 * label_strings filled with unique label names (same for train and test).
 * Caller must free train and test via free_classification_dataset(). */
void load_classification_dataset(const char* data_path, const char* labels_path,
                                  double train_percent, bool timeseries,
                                  ClassificationDataset* train, ClassificationDataset* test,
                                  std::vector<std::string>& label_strings);

/* Low-level classification: load a single dataset (no split), then split separately.
 * Returns {ClassificationDataset, label_strings}. */
std::pair<ClassificationDataset, std::vector<std::string>> load_classification_single(
    const char* data_path, const char* labels_path, bool timeseries);

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

/* Regression dataset loading */
void load_regression_dataset(const char* data_path, const char* target_path,
                              double train_percent, bool timeseries,
                              RegressionDataset* train, RegressionDataset* test);

void load_regression_dataset_single(const char* data_path, const char* target_path,
                                     bool timeseries,
                                     RegressionDataset* out);

void free_regression_dataset(RegressionDataset* ds);
