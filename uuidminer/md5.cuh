/*
 * md5.cuh CUDA Implementation of MD5 digest
 *
 * Date: 12 June 2019
 * Revision: 1
 *
 * Based on the public domain Reference Implementation in C, by
 * Brad Conte, original code here:
 *
 * https://github.com/B-Con/crypto-algorithms
 *
 * This file is released into the Public Domain.
 *
 * Modifications by: catme0w, 2024-11-08
 */


#pragma once
#include "common.h"

typedef struct
{
    u8 data[64];
    u32 datalen;
    unsigned long long bitlen;
    u32 state[4];
} cuda_md5_ctx;

__device__ void cuda_md5_init(cuda_md5_ctx* ctx);
__device__ void cuda_md5_update(cuda_md5_ctx* ctx, const u8 data[], size_t len);
__device__ void cuda_md5_final(cuda_md5_ctx* ctx, u8 hash[]);
void mcm_cuda_md5_hash_batch(u8* in, u32 inlen, u8* out, u32 n_batch);
