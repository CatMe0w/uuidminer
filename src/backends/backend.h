#pragma once
#include "../common/config.h"

class IBackend {
public:
    virtual ~IBackend() = default;
    virtual bool init(const Config& config) = 0;
    virtual void run() = 0;
};
