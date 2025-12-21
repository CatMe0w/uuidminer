#include "metal_backend.h"
#include "../../common/common.h"
#include "metal_kernel_source.h"
#include <algorithm>
#include <chrono>
#include <iostream>

// constants from cuda kernel
static constexpr u32 kAvailableCharLength = 63;
static constexpr u32 kAvailableCharLengthPow3 = 63 * 63 * 63;
static constexpr int kPlayerNameMaxLength = 16;

// best-effort NSError printer for debugging Metal underlying issues
static void printNSError(NSError* err)
{
    if (!err) {
        fprintf(stderr, "Metal error: (nil)\n");
        return;
    }

    fprintf(stderr, "Metal error domain=%s code=%ld\n",
        [[err domain] UTF8String], (long)[err code]);
    fprintf(stderr, "Metal error description=%s\n",
        [[err localizedDescription] UTF8String]);

    NSDictionary* userInfo = [err userInfo];
    if (userInfo) {
        fprintf(stderr, "Metal error userInfo keys: ");
        for (id k in userInfo) {
            NSString* ks = [k description];
            fprintf(stderr, "%s ", [ks UTF8String]);
        }
        fprintf(stderr, "\n");

        NSError* underlying = userInfo[NSUnderlyingErrorKey];
        if (underlying) {
            fprintf(stderr, "Underlying error:\n");
            fprintf(stderr, "  domain=%s code=%ld\n",
                [[underlying domain] UTF8String], (long)[underlying code]);
            fprintf(stderr, "  description=%s\n",
                [[underlying localizedDescription] UTF8String]);
        }
    }
}

MetalBackend::MetalBackend()
{
}

MetalBackend::~MetalBackend()
{
}

bool MetalBackend::init(const Config& config)
{
    _config = config;

    _device = MTLCreateSystemDefaultDevice();
    if (!_device) {
        std::cerr << "Error: No Metal device found." << std::endl;
        return false;
    }

    std::cout << "Using Metal device: " << [_device.name UTF8String] << std::endl;

    _commandQueue = [_device newCommandQueue];
    if (!_commandQueue) {
        std::cerr << "Error: Could not create command queue." << std::endl;
        return false;
    }

    NSError* error = nil;
    MTLCompileOptions* options = [MTLCompileOptions new];

    if (@available(macOS 15.0, iOS 18.0, *)) {
        options.mathMode = MTLMathModeFast;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = YES;
#pragma clang diagnostic pop
    }

    std::string src = kMetalKernelSource;

    NSString* source = [NSString stringWithUTF8String:src.c_str()];
    id<MTLLibrary> library = [_device newLibraryWithSource:source options:options error:&error];

    if (!library || error) {
        std::cerr << "Error: Failed to compile Metal library: "
                  << (error ? [[error localizedDescription] UTF8String] : "unknown")
                  << std::endl;
        return false;
    }

    _pipelineStates.clear();
    _pipelineStates.reserve(kPlayerNameMaxLength - 2);

    for (int i = 0; i <= kPlayerNameMaxLength - 3; ++i) {
        std::string fn = "md5_hash_player_name_" + std::to_string(i);
        id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:fn.c_str()]];

        if (!function) {
            std::cerr << "Error: Failed to find Metal function: " << fn << std::endl;
            return false;
        }

        NSError* pso_error = nil;
        id<MTLComputePipelineState> pso = [_device newComputePipelineStateWithFunction:function error:&pso_error];

        if (!pso || pso_error) {
            std::cerr << "Error: Failed to create pipeline for " << fn << ": "
                      << (pso_error ? [[pso_error localizedDescription] UTF8String] : "unknown")
                      << std::endl;
            return false;
        }

        _pipelineStates.push_back(pso);
    }

    fprintf(stderr, "Node configuration: Index %d / %d (Slices: %d)\n",
        _config.node_index, _config.node_count, _config.node_slices);
    fprintf(stderr, "Target prefix: %s\n\n", _config.target_str.c_str());

    return true;
}

void MetalBackend::run()
{
    @autoreleasepool {
        struct Params {
            u32 start;
            u32 end;
            u32 target_state0;
            u32 _pad;
            u64 iter_start;
            u64 iter_end;
        };

        struct Result {
            u32 name_len;
            u8 name[kPlayerNameMaxLength];
            u32 _pad0;
            u64 final_hi;
            u64 final_lo;
        };

        static constexpr u32 kMaxResults = 1024;

        for (int i = 0; i <= 13; ++i) {
            fprintf(stderr, "Searching %d-character player names...\n", i + 3);

            auto startTime = std::chrono::high_resolution_clock::now();

            constexpr u32 threads_per_threadgroup = 256;
            constexpr u32 total_global_threads = kAvailableCharLengthPow3;

            // calculate node range
            const u32 node_chunk_size = (total_global_threads + _config.node_count - 1) / _config.node_count;
            const u32 node_start = _config.node_index * node_chunk_size;
            const u32 node_end = std::min(node_start + node_chunk_size * _config.node_slices, total_global_threads);

            if (node_start >= node_end) {
                fprintf(stderr, "Empty node slice; nothing to do.\n\n");
                continue;
            }

            // assume always single GPU on Mac: one device range equals node range
            const u32 start = node_start;
            const u32 end = node_end;
            const u32 count = end - start;

            // use dispatchThreads with rounded-up thread count; kernel guards by end
            const u32 global_threads = ((count + threads_per_threadgroup - 1) / threads_per_threadgroup) * threads_per_threadgroup;

            // split traversal space into chunks to avoid macOS GPU "Impacting Interactivity" aborts
            auto pow63 = [](int exp) -> u64 {
                u64 v = 1;
                for (int j = 0; j < exp; ++j)
                    v *= (u64)kAvailableCharLength;
                return v;
            };

            const u64 total_iters = pow63(i);
            // keep each command buffer short to avoid macOS GPU watchdog ("Impacting Interactivity")
            // for capturing watchdog samples, pass --metal-no-chunking
            u64 iter_chunk = 0;
            if (_config.metal_no_chunking) {
                iter_chunk = total_iters;
            } else {
                // i==3 (6-char) is the heavy case: 63^3 threads * iter_chunk iterations per thread
                // i.e. 250 million iterations per command buffer => ~13s on M1 Pro
                // starting from here, "Impacting Interactivity" may occur if the screen is active,
                // leading to (silent) command buffer aborts and incomplete results
                if (i <= 2)
                    iter_chunk = total_iters;
                else
                    iter_chunk = (_config.metal_iter_chunk == 0 ? total_iters : (u64)_config.metal_iter_chunk);
            }

            id<MTLBuffer> matchCountBuffer = [_device newBufferWithLength:sizeof(u32)
                                                                  options:MTLResourceStorageModeShared];
            id<MTLBuffer> resultsBuffer = [_device newBufferWithLength:sizeof(Result) * kMaxResults
                                                               options:MTLResourceStorageModeShared];

            *reinterpret_cast<u32*>([matchCountBuffer contents]) = 0;

            u32 printed = 0;
            bool warned_overflow = false;

            MTLSize threadsPerThreadgroup = MTLSizeMake(threads_per_threadgroup, 1, 1);
            MTLSize threadsPerGrid = MTLSizeMake(global_threads, 1, 1);

            for (u64 iter_start64 = 0; iter_start64 < total_iters; iter_start64 += iter_chunk) {
                const u64 iter_end64 = std::min(iter_start64 + iter_chunk, total_iters);
                Params p { start, end, _config.target_state0, 0, iter_start64, iter_end64 };

                id<MTLCommandBuffer> commandBuffer = nil;
                if (@available(macOS 11.0, iOS 14.0, *)) {
                    // request richer error info when available
                    MTLCommandBufferDescriptor* desc = [MTLCommandBufferDescriptor new];
                    desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
                    commandBuffer = [_commandQueue commandBufferWithDescriptor:desc];
                } else {
                    commandBuffer = [_commandQueue commandBuffer];
                }

                commandBuffer.label = [NSString stringWithFormat:@"uuidminer_len%d_iter%llu_%llu", i + 3,
                    (unsigned long long)iter_start64, (unsigned long long)iter_end64];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

                [encoder setComputePipelineState:_pipelineStates[i]];
                [encoder setBytes:&p length:sizeof(Params) atIndex:0];
                [encoder setBuffer:matchCountBuffer offset:0 atIndex:1];
                [encoder setBuffer:resultsBuffer offset:0 atIndex:2];

                [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
                [encoder endEncoding];

                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];

                if (commandBuffer.status == MTLCommandBufferStatusError) {
                    fprintf(stderr, "Metal command buffer failed (status=Error) label=%s\n",
                        [[commandBuffer.label description] UTF8String]);
                    printNSError(commandBuffer.error);
                    return;
                }

                // print newly found results incrementally so output appears quickly
                const u32 matchCount = *reinterpret_cast<u32*>([matchCountBuffer contents]);
                const u32 toPrintNow = std::min(matchCount, kMaxResults);

                if (toPrintNow > printed) {
                    Result* results = reinterpret_cast<Result*>([resultsBuffer contents]);

                    for (u32 r = printed; r < toPrintNow; ++r) {
                        char nameBuf[kPlayerNameMaxLength + 1];
                        const u32 len = std::min(results[r].name_len, (u32)kPlayerNameMaxLength);

                        for (u32 k = 0; k < len; ++k)
                            nameBuf[k] = (char)results[r].name[k];
                        nameBuf[len] = '\0';

                        const u64 final_hi = results[r].final_hi;
                        const u64 final_lo = results[r].final_lo;

                        printf("%s,%08x-%04x-%04x-%04x-%04x%08x\n",
                            nameBuf,
                            (u32)(final_hi >> 32),
                            (u32)((final_hi >> 16) & 0xFFFF),
                            (u32)(final_hi & 0xFFFF),
                            (u32)((final_lo >> 48) & 0xFFFF),
                            (u32)((final_lo >> 32) & 0xFFFF),
                            (u32)(final_lo & 0xFFFFFFFF));
                    }

                    printed = toPrintNow;
                    fflush(stdout);
                }

                if (!warned_overflow && matchCount > kMaxResults) {
                    fprintf(stderr, "(Metal) Warning: match buffer overflow: %u (printing capped at %u)\n", matchCount, kMaxResults);
                    warned_overflow = true;
                }
            }

            auto endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = endTime - startTime;
            fprintf(stderr, "Time elapsed: %.3f s\n\n", elapsed.count());
        }
    }
}
