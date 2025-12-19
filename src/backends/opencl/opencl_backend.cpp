#include "opencl_backend.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>

#include "../../common/common.h"

// constants from cuda kernel
constexpr auto available_char_length = 63;
constexpr auto available_char_length_pow_3 = 63 * 63 * 63;
constexpr auto player_name_max_length = 16;

OpenCLBackend::OpenCLBackend() = default;

OpenCLBackend::~OpenCLBackend()
{
#ifdef USE_OPENCL
    for (auto& dev : m_devices)
    {
        for (auto& k : dev.kernels)
        {
            if (k) clReleaseKernel(k);
        }
        if (dev.program) clReleaseProgram(dev.program);
        if (dev.command_queue) clReleaseCommandQueue(dev.command_queue);
        if (dev.context) clReleaseContext(dev.context);
    }
#endif
}

#ifdef USE_OPENCL
const char* cl_get_error_string(cl_int err)
{
    switch (err)
    {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
    default: return "Unknown OpenCL error";
    }
}

void OpenCLBackend::check_error(cl_int err, const char* operation)
{
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "OpenCL Error during %s: %s\n", operation, cl_get_error_string(err));
    }
}

bool OpenCLBackend::load_kernel_source(const std::string& filename, std::string& source)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        fprintf(stderr, "Failed to open kernel file: %s\n", filename.c_str());
        return false;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    source = buffer.str();
    return true;
}
#endif

bool OpenCLBackend::init(const Config& config)
{
    m_config = config;

#ifndef USE_OPENCL
    fprintf(stderr, "OpenCL support not compiled in.\n");
    return false;
#else
    cl_int err;
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS)
    {
        check_error(err, "clGetPlatformIDs");
        return false;
    }
    if (num_platforms == 0)
    {
        fprintf(stderr, "No OpenCL platforms found.\n");
        return false;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    cl_platform_id platform = platforms[0];

    cl_uint num_devices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (err != CL_SUCCESS)
    {
        check_error(err, "clGetDeviceIDs");
        return false;
    }
    if (num_devices == 0)
    {
        fprintf(stderr, "No OpenCL GPU devices found.\n");
        return false;
    }

    std::vector<cl_device_id> device_ids(num_devices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, device_ids.data(), nullptr);
    fprintf(stderr, "Found %d OpenCL device(s)\n", num_devices);

    std::string kernel_source;
    if (!load_kernel_source("src/backends/opencl/kernel.cl", kernel_source) &&
        !load_kernel_source("../src/backends/opencl/kernel.cl", kernel_source))
    {
        return false;
    }

    const char* source_str = kernel_source.c_str();
    size_t source_size = kernel_source.size();

    size_t size;
    for (cl_device_id dev_id : device_ids)
    {
        DeviceContext dev_ctx;
        dev_ctx.device_id = dev_id;

        clGetDeviceInfo(dev_id, CL_DEVICE_NAME, 0, nullptr, &size);
        dev_ctx.device_name.resize(size);
        clGetDeviceInfo(dev_id, CL_DEVICE_NAME, size, &dev_ctx.device_name[0], nullptr);
        if (!dev_ctx.device_name.empty() && dev_ctx.device_name.back() == '\0')
            dev_ctx.device_name.pop_back();
        fprintf(stderr, "Device: %s\n", dev_ctx.device_name.c_str());

        dev_ctx.context = clCreateContext(nullptr, 1, &dev_id, nullptr, nullptr, &err);
        if (err != CL_SUCCESS)
        {
            check_error(err, "clCreateContext");
            return false;
        }

#ifdef CL_VERSION_2_0
        dev_ctx.command_queue = clCreateCommandQueueWithProperties(dev_ctx.context, dev_id, 0, &err);
#else
        dev_ctx.command_queue = clCreateCommandQueue(dev_ctx.context, dev_id, 0, &err);
#endif
        if (err != CL_SUCCESS)
        {
            check_error(err, "clCreateCommandQueue");
            return false;
        }

        dev_ctx.program = clCreateProgramWithSource(dev_ctx.context, 1, &source_str, &source_size, &err);
        if (err != CL_SUCCESS)
        {
            check_error(err, "clCreateProgramWithSource");
            return false;
        }
        err = clBuildProgram(dev_ctx.program, 1, &dev_id, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            size_t log_size;
            clGetProgramBuildInfo(dev_ctx.program, dev_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(dev_ctx.program, dev_id, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
            fprintf(stderr, "OpenCL Program Build Error:\n%s\n", log.c_str());
            return false;
        }

        for (int i = 0; i <= player_name_max_length - 3; ++i)
        {
            std::string kernel_name = "md5_hash_player_name_" + std::to_string(i);
            cl_kernel k = clCreateKernel(dev_ctx.program, kernel_name.c_str(), &err);
            if (err != CL_SUCCESS)
            {
                check_error(err, ("clCreateKernel " + kernel_name).c_str());
                return false;
            }
            dev_ctx.kernels.push_back(k);
        }

        m_devices.push_back(dev_ctx);
    }

    fprintf(stderr, "Node configuration: Index %d / %d (Slices: %d)\n",
            m_config.node_index, m_config.node_count, m_config.node_slices);
    fprintf(stderr, "Target prefix: %s\n\n", m_config.target_str.c_str());

    return true;
#endif
}

void OpenCLBackend::run()
{
#ifdef USE_OPENCL
    for (int i = 0; i <= player_name_max_length - 3; ++i)
    {
        fprintf(stderr, "Searching %d-character player names...\n", i + 3);

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        int device_count = m_devices.size();

        for (int d = 0; d < device_count; ++d)
        {
            threads.emplace_back([d, i, device_count, this]()
            {
                DeviceContext& dev = m_devices[d];
                
                constexpr u32 threads_per_block = 256;
                constexpr u32 total_global_threads = available_char_length_pow_3;

                // Calculate node range
                const u32 node_chunk_size = (total_global_threads + m_config.node_count - 1) / m_config.node_count;
                const u32 node_start = m_config.node_index * node_chunk_size;
                const u32 node_end = std::min(node_start + node_chunk_size * m_config.node_slices, total_global_threads);

                if (node_start >= node_end) return;

                const u32 node_total_threads = node_end - node_start;

                // Calculate device range within node
                const u32 device_chunk_size = (node_total_threads + device_count - 1) / device_count;
                const u32 device_start_offset = d * device_chunk_size;
                const u32 device_end_offset = std::min(device_start_offset + device_chunk_size, node_total_threads);

                if (device_start_offset >= device_end_offset) return;

                const u32 start = node_start + device_start_offset;
                const u32 count = device_end_offset - device_start_offset;
                
                size_t global_work_size = ((count + threads_per_block - 1) / threads_per_block) * threads_per_block;
                size_t local_work_size = threads_per_block;

                cl_kernel kernel = dev.kernels[i];
                cl_int err;
                
                err = clSetKernelArg(kernel, 0, sizeof(u32), &start);
                check_error(err, "clSetKernelArg 0");

                err = clSetKernelArg(kernel, 1, sizeof(u32), &m_config.target_state0);
                check_error(err, "clSetKernelArg 1");

                err = clEnqueueNDRangeKernel(dev.command_queue, kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
                check_error(err, "clEnqueueNDRangeKernel");
                
                err = clFinish(dev.command_queue);
                check_error(err, "clFinish");
            });
        }

        for (auto& t : threads)
        {
            if (t.joinable()) t.join();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        fprintf(stderr, "Time elapsed: %.3f s\n\n", elapsed.count());
    }
#endif
}
