#include "backend.h"
#include "cpu_backend.h"
#include "opencl_backend.h"
#include <cstdio>
#include <cstdlib>
#include <memory>

std::unique_ptr<TrainingBackend> create_backend(
    const CliConfig& cfg,
    neuro::Network* n,
    NetworkConfiguration& nc,
    const Dataset& train,
    const Dataset& test,
    TrainingState* state,
    size_t batch_size,
    double learning_rate,
    double decay_rate,
    double rho,
    double tau)
{
    if (cfg.opencl) {
        if (!nc.discrete) {
            fprintf(stderr,
                "OpenCL support is not enabled for non-discrete networks.\n");
            exit(1);
        }
        return std::unique_ptr<TrainingBackend>(
            new OpenclBackend(
                cfg, n, nc, train, test, state,
                nc.max_incoming, nc.max_outgoing,
                batch_size, learning_rate, decay_rate,
                rho, tau, true));
    } else {
        return std::unique_ptr<TrainingBackend>(
            new CpuBackend(
                cfg, n, nc, train, test, state,
                batch_size, learning_rate, decay_rate,
                !cfg.network_json_out.empty()));
    }
}
