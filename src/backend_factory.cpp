#include "backend.h"
#include "cpu_backend.h"
#ifdef OPENCL
#include "opencl_backend.h"
#endif
#include <cstdio>
#include <cstdlib>
#include <memory>

std::unique_ptr<TrainingBackend> create_backend(const CliConfig& cfg,
                                                NetworkConfiguration& nc,
                                                const Dataset& train,
                                                const Dataset& test) {
#ifdef OPENCL
    if (cfg.opencl) {
        if (!nc.discrete) {
            fprintf(
                stderr,
                "OpenCL support is not enabled for non-discrete networks.\n");
            exit(1);
        }
        return std::unique_ptr<TrainingBackend>(new OpenclBackend(
            cfg, nc, train, test, nc.max_incoming, nc.max_outgoing));
    }
#endif
    return std::unique_ptr<TrainingBackend>(
        new CpuBackend(cfg, nc, train, test));
}
