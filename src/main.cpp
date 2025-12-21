#include "backends/backend.h"
#include "common/config.h"
#include <memory>

#ifdef USE_CUDA
#include "backends/cuda/cuda_backend.h"
#endif

#ifdef USE_METAL
#include "backends/metal/metal_backend.h"
#endif

#ifdef USE_OPENCL
#include "backends/opencl/opencl_backend.h"
#endif

int main(int argc, char** argv)
{
    Config config;
    if (!Config::parse(argc, argv, config))
    {
        return 1;
    }

    std::unique_ptr<IBackend> backend;

    if (config.backend == "cuda")
    {
#ifdef USE_CUDA
        backend = std::make_unique<CudaBackend>();
#else
        fprintf(stderr, "CUDA backend not available (compiled without USE_CUDA)\n");
        return 1;
#endif
    }
    else if (config.backend == "metal")
    {
#ifdef USE_METAL
        backend = std::make_unique<MetalBackend>();
#else
        fprintf(stderr, "Metal backend not available (compiled without USE_METAL)\n");
        return 1;
#endif
    }
    else if (config.backend == "opencl")
    {
#ifdef USE_OPENCL
        backend = std::make_unique<OpenCLBackend>();
#else
        fprintf(stderr, "OpenCL backend not available (compiled without USE_OPENCL)\n");
        return 1;
#endif
    }
    else
    {
        fprintf(stderr, "Unknown backend: %s\n", config.backend.c_str());
        return 1;
    }

    if (!backend->init(config))
    {
        fprintf(stderr, "Failed to initialize backend\n");
        return 1;
    }

    backend->run();

    return 0;
}
