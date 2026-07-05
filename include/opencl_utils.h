#ifndef OPENCL_UTILS_H
#define OPENCL_UTILS_H

#include <CL/cl.h>
#include <cstddef>

char* load_kernel(const char* path);
void opencl_perror(int err);
void opencl_zerobuf(cl_command_queue q, void* buf, size_t n, size_t sizeper,
                    cl_event* event);

#endif // OPENCL_UTILS_H
