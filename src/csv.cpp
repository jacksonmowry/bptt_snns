#include "csv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Dataset load_dataset(const char* data_path, const char* labels_path) {
    Dataset ds = {NULL, NULL, NULL, NULL, 0, 0};
    FILE* f_data = fopen(data_path, "r");
    FILE* f_labels = fopen(labels_path, "r");
    if (f_data == NULL || f_labels == NULL) {
        if (f_data != NULL)
            fclose(f_data);
        if (f_labels != NULL)
            fclose(f_labels);
        return ds;
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

    ds.data = (double*)malloc(rows * cols * sizeof(double));
    ds.labels = (double*)malloc(rows * sizeof(double));
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
        return ds;
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
                    token = strtok(NULL, ",");
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
    return ds;
}
