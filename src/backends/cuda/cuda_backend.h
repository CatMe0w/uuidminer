#pragma once
#include "../../common/config.h"
#include "../backend.h"

class CudaBackend : public IBackend
{
  public:
    CudaBackend();
    ~CudaBackend() override;

    bool init(const Config& config) override;
    void run() override;

  private:
    Config m_config;
    int m_device_count = 0;
};
