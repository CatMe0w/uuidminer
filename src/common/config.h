#pragma once
#include "common.h"
#include <iostream>
#include <string>
#include <vector>

struct Config
{
    int node_index = 0;
    int node_count = 1;
    int node_slices = 1;
    std::string target_str = "00000000";
    u32 target_state0 = 0;
#ifdef USE_CUDA
    std::string backend = "cuda"; // Default to cuda
#elif defined(USE_OPENCL)
    std::string backend = "opencl"; // Default to opencl
#else
    std::string backend = ""; // No default backend; XXX: add CPU backend
#endif

    static bool parse(const int argc, char** argv, Config& config)
    {
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg == "--node-index" && i + 1 < argc)
            {
                config.node_index = std::stoi(argv[++i]);
            }
            else if (arg == "--node-count" && i + 1 < argc)
            {
                config.node_count = std::stoi(argv[++i]);
            }
            else if (arg == "--node-slices" && i + 1 < argc)
            {
                config.node_slices = std::stoi(argv[++i]);
            }
            else if (arg == "--target" && i + 1 < argc)
            {
                config.target_str = argv[++i];
            }
            else if (arg == "--backend" && i + 1 < argc)
            {
                config.backend = argv[++i];
            }
        }

        if (config.target_str.length() != 8)
        {
            fprintf(stderr, "Invalid target length: %s (must be 8 hex chars)\n\n", config.target_str.c_str());
            return false;
        }

        u32 target_val;
        try
        {
            target_val = std::stoul(config.target_str, nullptr, 16);
        }
        catch (...)
        {
            fprintf(stderr, "Invalid target hex: %s\n", config.target_str.c_str());
            return false;
        }

        if (config.node_index < 0 || config.node_index >= config.node_count)
        {
            fprintf(stderr, "Invalid node configuration: index %d, count %d\n\n", config.node_index, config.node_count);
            return false;
        }

        if (config.node_slices <= 0 || config.node_index + config.node_slices > config.node_count)
        {
            fprintf(stderr,
                    "Invalid node slices configuration: index %d, slices %d, count %d (index + slices must be <= "
                    "count)\n\n",
                    config.node_index, config.node_slices, config.node_count);
            return false;
        }

        // convert target_val (big-endian) to target_state0 (little-endian)
        config.target_state0 = ((target_val & 0xFF) << 24) | ((target_val & 0xFF00) << 8) |
                               ((target_val & 0xFF0000) >> 8) | ((target_val >> 24) & 0xFF);

        return true;
    }
};
