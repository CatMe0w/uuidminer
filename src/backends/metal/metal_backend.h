#pragma once
#include "../../common/config.h"
#include "../backend.h"

#include <vector>

#ifdef __OBJC__
#import <Metal/Metal.h>
#else
// forward declare Metal types for non-Objective-C compilation units
typedef void* id;
#endif

class MetalBackend : public IBackend {
public:
    MetalBackend();
    ~MetalBackend() override;

    bool init(const Config& config) override;
    void run() override;

private:
    Config _config;

#ifdef __OBJC__
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    std::vector<id<MTLComputePipelineState>> _pipelineStates;
#else
    void* _device = nullptr;
    void* _commandQueue = nullptr;
    std::vector<void*> _pipelineStates;
#endif
};
