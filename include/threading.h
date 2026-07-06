#pragma once

#include <pthread.h>
#include <cstddef>
#include <vector>
#include "csv.h"
#include "shared.h"
#include "framework.hpp"

struct ThreadArgs {
    TrainingBundle tb;
    NetworkConfiguration* nc;
    const size_t* order;

    const Dataset* train;
    const Dataset* test;
    int* max_idx;
    int* work_idx;
    int* done;

    double loss      = 0;
    size_t correct   = 0;
    size_t processed = 0;

    pthread_mutex_t* mut;
    pthread_cond_t* have_work;
    pthread_cond_t* done_work;
    bool* train_p;
    bool* die;

    ThreadArgs(size_t total_neurons, size_t timesteps, size_t output_neurons,
               double rho, double tau, const std::vector<std::vector<double>>* weights,
               const std::vector<std::vector<int>>* delays,
               const std::vector<double>* thresholds, NetworkConfiguration* nc,
               const size_t* order, const Dataset* train, const Dataset* test,
               int* max_idx, int* work_idx, int* done, pthread_mutex_t* mut,
               pthread_cond_t* have_work, pthread_cond_t* done_work,
               bool* train_p, bool* die);
};

void* worker(void* arg);
