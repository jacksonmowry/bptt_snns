#include "threading.h"
#include "forward_backward.h"
#include "framework.hpp"
#include "network_utils.h"
#include "shared.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>

ThreadArgs::ThreadArgs(size_t total_neurons, size_t timesteps,
                       size_t output_neurons, double rho, double tau,
                       const std::vector<std::vector<double>>* weights,
                       const std::vector<std::vector<int>>* delays,
                       const std::vector<double>* thresholds,
                       NetworkConfiguration* nc, const size_t* order,
                       const Dataset* train, const Dataset* test, int* max_idx,
                       int* work_idx, int* done, pthread_mutex_t* mut,
                       pthread_cond_t* have_work, pthread_cond_t* done_work,
                       bool* train_p, bool* die)
    : tb(total_neurons, timesteps, output_neurons, rho, tau, weights, delays,
         thresholds),
      nc(nc), order(order), train(train), test(test), max_idx(max_idx),
      work_idx(work_idx), done(done), mut(mut), have_work(have_work),
      done_work(done_work), train_p(train_p), die(die) {}

void* worker(void* arg) {
    if (!arg) {
        fprintf(stderr, "Thread spawned with NULL arg\n");
        exit(1);
    }

    ThreadArgs* ta         = (ThreadArgs*)arg;
    neuro::Processor* p    = NULL;
    bool do_backwards_pass = false;
    int my_max;

    while (true) {
        pthread_mutex_lock(ta->mut);
        if (*ta->die) {
            pthread_mutex_unlock(ta->mut);
            pthread_exit(NULL);
        }

        while (*ta->work_idx >= *ta->max_idx) {
            pthread_cond_wait(ta->have_work, ta->mut);
            if (*ta->die) {
                pthread_mutex_unlock(ta->mut);
                pthread_exit(NULL);
            }
        }
        int my_work_idx   = *ta->work_idx;
        *ta->work_idx     = my_work_idx + 1;
        my_max            = *ta->max_idx;
        do_backwards_pass = *ta->train_p;
        pthread_mutex_unlock(ta->mut);

        load_network(&p, ta->nc->n);

        while (my_work_idx < my_max) {
            EvaluationResults er = forward(
                &ta->tb, p, *ta->train_p ? ta->train : ta->test,
                *ta->train_p ? ta->order[my_work_idx] : my_work_idx, ta->nc);
            ta->loss += er.loss;
            ta->correct += er.correct;
            ta->processed++;

            if (do_backwards_pass) {
                backward(&ta->tb, ta->nc);
            }

            pthread_mutex_lock(ta->mut);
            my_work_idx   = *ta->work_idx;
            *ta->work_idx = my_work_idx + 1;
            assert(*ta->max_idx == my_max);
            do_backwards_pass = *ta->train_p;
            pthread_mutex_unlock(ta->mut);
        }

        pthread_mutex_lock(ta->mut);
        *ta->done = *ta->done + ta->processed;
        pthread_cond_signal(ta->done_work);
        pthread_mutex_unlock(ta->mut);
        delete p;
        p = NULL;
    }

    return NULL;
}
