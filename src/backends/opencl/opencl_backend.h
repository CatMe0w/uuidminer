#pragma once
#include "../../common/config.h"
#include "../backend.h"

#include <string>
#include <vector>

#ifdef USE_OPENCL
#if defined(__APPLE__)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#endif

class OpenCLBackend : public IBackend
{
  public:
    OpenCLBackend();
    ~OpenCLBackend() override;

    bool init(const Config& config) override;
    void run() override;

  private:
    Config m_config;
#ifdef USE_OPENCL
    struct DeviceContext
    {
        cl_device_id device_id;
        cl_context context;
        cl_command_queue command_queue;
        cl_program program;
        std::vector<cl_kernel> kernels; // Store kernels for each length
        std::string device_name;
    };

    std::vector<DeviceContext> m_devices;

    bool load_kernel_source(const std::string& filename, std::string& source);
    void check_error(cl_int err, const char* operation);
#endif
};
