#include "opencl_utils.h"
#include "backtrace.h"
#include <cstdio>
#include <cstdlib>

char* load_kernel(const char* path) {
    FILE* f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "Unable to open kernel for reading: %s\n", path);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* src = (char*)malloc(len + 1);
    fread(src, 1, len, f);
    src[len] = '\0';
    fclose(f);
    return src;
}

void opencl_perror(int err) {
    switch (err) {
    case 0:
        puts("CL_SUCCESS");
        break;
    case -1:
        puts("CL_DEVICE_NOT_FOUND                        ");
        break;
    case -2:
        puts("CL_DEVICE_NOT_AVAILABLE                    ");
        break;
    case -3:
        puts("CL_COMPILER_NOT_AVAILABLE                  ");
        break;
    case -4:
        puts("CL_MEM_OBJECT_ALLOCATION_FAILURE           ");
        break;
    case -5:
        puts("CL_OUT_OF_RESOURCES                        ");
        break;
    case -6:
        puts("CL_OUT_OF_HOST_MEMORY                      ");
        break;
    case -7:
        puts("CL_PROFILING_INFO_NOT_AVAILABLE            ");
        break;
    case -8:
        puts("CL_MEM_COPY_OVERLAP                        ");
        break;
    case -9:
        puts("CL_IMAGE_FORMAT_MISMATCH                   ");
        break;
    case -10:
        puts("CL_IMAGE_FORMAT_NOT_SUPPORTED              ");
        break;
    case -11:
        puts("CL_BUILD_PROGRAM_FAILURE                   ");
        break;
    case -12:
        puts("CL_MAP_FAILURE                             ");
        break;
    case -13:
        puts("CL_MISALIGNED_SUB_BUFFER_OFFSET            ");
        break;
    case -14:
        puts("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST");
        break;
    case -15:
        puts("CL_COMPILE_PROGRAM_FAILURE");
        break;
    case -16:
        puts("CL_LINKER_NOT_AVAILABLE                    ");
        break;
    case -17:
        puts("CL_LINK_PROGRAM_FAILURE                    ");
        break;
    case -18:
        puts("CL_DEVICE_PARTITION_FAILED                 ");
        break;
    case -19:
        puts("CL_KERNEL_ARG_INFO_NOT_AVAILABLE           ");
        break;
    case -30:
        puts("CL_INVALID_VALUE                           ");
        break;
    case -31:
        puts("CL_INVALID_DEVICE_TYPE                     ");
        break;
    case -32:
        puts("CL_INVALID_PLATFORM                        ");
        break;
    case -33:
        puts("CL_INVALID_DEVICE                          ");
        break;
    case -34:
        puts("CL_INVALID_CONTEXT                         ");
        break;
    case -35:
        puts("CL_INVALID_QUEUE_PROPERTIES                ");
        break;
    case -36:
        puts("CL_INVALID_COMMAND_QUEUE                   ");
        break;
    case -37:
        puts("CL_INVALID_HOST_PTR                        ");
        break;
    case -38:
        puts("CL_INVALID_MEM_OBJECT                      ");
        break;
    case -39:
        puts("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR         ");
        break;
    case -40:
        puts("CL_INVALID_IMAGE_SIZE                      ");
        break;
    case -41:
        puts("CL_INVALID_SAMPLER                         ");
        break;
    case -42:
        puts("CL_INVALID_BINARY                          ");
        break;
    case -43:
        puts("CL_INVALID_BUILD_OPTIONS                   ");
        break;
    case -44:
        puts("CL_INVALID_PROGRAM                         ");
        break;
    case -45:
        puts("CL_INVALID_PROGRAM_EXECUTABLE              ");
        break;
    case -46:
        puts("CL_INVALID_KERNEL_NAME                     ");
        break;
    case -47:
        puts("CL_INVALID_KERNEL_DEFINITION               ");
        break;
    case -48:
        puts("CL_INVALID_KERNEL                          ");
        break;
    case -49:
        puts("CL_INVALID_ARG_INDEX                       ");
        break;
    case -50:
        puts("CL_INVALID_ARG_VALUE                       ");
        break;
    case -51:
        puts("CL_INVALID_ARG_SIZE                        ");
        break;
    case -52:
        puts("CL_INVALID_KERNEL_ARGS                     ");
        break;
    case -53:
        puts("CL_INVALID_WORK_DIMENSION                  ");
        break;
    case -54:
        puts("CL_INVALID_WORK_GROUP_SIZE                 ");
        break;
    case -55:
        puts("CL_INVALID_WORK_ITEM_SIZE                  ");
        break;
    case -56:
        puts("CL_INVALID_GLOBAL_OFFSET                   ");
        break;
    case -57:
        puts("CL_INVALID_EVENT_WAIT_LIST                 ");
        break;
    case -58:
        puts("CL_INVALID_EVENT                           ");
        break;
    case -59:
        puts("CL_INVALID_OPERATION                       ");
        break;
    case -60:
        puts("CL_INVALID_GL_OBJECT                       ");
        break;
    case -61:
        puts("CL_INVALID_BUFFER_SIZE                     ");
        break;
    case -62:
        puts("CL_INVALID_MIP_LEVEL                       ");
        break;
    case -63:
        puts("CL_INVALID_GLOBAL_WORK_SIZE                ");
        break;
    case -64:
        puts("CL_INVALID_PROPERTY                        ");
        break;
    case -65:
        puts("CL_INVALID_IMAGE_DESCRIPTOR                ");
        break;
    case -66:
        puts("CL_INVALID_COMPILER_OPTIONS                ");
        break;
    case -67:
        puts("CL_INVALID_LINKER_OPTIONS                  ");
        break;
    case -68:
        puts("CL_INVALID_DEVICE_PARTITION_COUNT          ");
        break;
    }
}

void opencl_zerobuf(cl_command_queue q, void* buf, size_t n, size_t sizeper,
                    cl_event* event) {
    size_t zero = 0;
    cl_event next_event;
    int err = clEnqueueFillBuffer(q, (cl_mem)buf, &zero, sizeper, 0,
                                  n * sizeper, (*event != NULL) ? 1 : 0,
                                  (*event != NULL) ? event : NULL, &next_event);

    if (err != CL_SUCCESS) {
        fprintf(stderr,
                "Error: clEnqueueFillBuffer failed with error code %d: ", err);
        opencl_perror(err);
        print_backtrace();
        exit(1);
    }

    *event = next_event;
    return;
}
