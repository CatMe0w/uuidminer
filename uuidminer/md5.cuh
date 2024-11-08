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
void mcm_cuda_md5_hash_batch(u8* in, u32 inlen, u8* out, u32 n_batch);
