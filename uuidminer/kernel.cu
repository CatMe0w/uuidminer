#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cmath>

#include "md5.cuh"

__constant__ u8 player_name_prefix[] = {
    'O', 'f', 'f', 'l', 'i', 'n', 'e', 'P', 'l', 'a', 'y', 'e', 'r', ':'
};

constexpr auto player_name_prefix_length = 14;

__constant__ u8 available_chars[] = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
    'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z',
    '_'
};

constexpr auto available_char_length = 63;
constexpr auto available_char_length_pow_2 = 63 * 63;
constexpr auto available_char_length_pow_3 = 63 * 63 * 63;

constexpr auto player_name_max_length = 16;

__device__ void get_next_index(u8* num, const int length)
{
    // base 63 integer
    int carry = 1;

    for (int i = length - 1; i >= 0 && carry > 0; --i)
    {
        num[i] += carry;
        if (num[i] >= 63)
        {
            num[i] = 0;
            carry = 1;
        }
        else
        {
            carry = 0;
        }
    }
}

__device__ void convert_md5_to_u128(const u8 md5[md5_block_size], u64* hi, u64* lo)
{
    *hi = (static_cast<u64>(md5[0]) << 56) | (static_cast<u64>(md5[1]) << 48) |
        (static_cast<u64>(md5[2]) << 40) | (static_cast<u64>(md5[3]) << 32) |
        (static_cast<u64>(md5[4]) << 24) | (static_cast<u64>(md5[5]) << 16) |
        (static_cast<u64>(md5[6]) << 8) | static_cast<u64>(md5[7]);

    *lo = (static_cast<u64>(md5[8]) << 56) | (static_cast<u64>(md5[9]) << 48) |
        (static_cast<u64>(md5[10]) << 40) | (static_cast<u64>(md5[11]) << 32) |
        (static_cast<u64>(md5[12]) << 24) | (static_cast<u64>(md5[13]) << 16) |
        (static_cast<u64>(md5[14]) << 8) | static_cast<u64>(md5[15]);
}

void convert_md5_to_u128_cpu(const u8 md5[md5_block_size], u64* hi, u64* lo)
{
    *hi = (static_cast<u64>(md5[0]) << 56) | (static_cast<u64>(md5[1]) << 48) |
        (static_cast<u64>(md5[2]) << 40) | (static_cast<u64>(md5[3]) << 32) |
        (static_cast<u64>(md5[4]) << 24) | (static_cast<u64>(md5[5]) << 16) |
        (static_cast<u64>(md5[6]) << 8) | static_cast<u64>(md5[7]);

    *lo = (static_cast<u64>(md5[8]) << 56) | (static_cast<u64>(md5[9]) << 48) |
        (static_cast<u64>(md5[10]) << 40) | (static_cast<u64>(md5[11]) << 32) |
        (static_cast<u64>(md5[12]) << 24) | (static_cast<u64>(md5[13]) << 16) |
        (static_cast<u64>(md5[14]) << 8) | static_cast<u64>(md5[15]);
}

__global__ void kernel_md5_hash_player_name(const int length, u8* cuda_indata, u8* cuda_outdata)
{
    u32 thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread >= available_char_length_pow_3)
    {
        return;
    }

    // pattern: [prefix][in_traversal_part][byte_a][byte_b][byte_c]
    // we are trying to find the best [in_traversal_part] that gives the smallest MD5 hash (offline uuid)
    // by iterating through all possible [in_traversal_part]
    //
    // prefix: "OfflinePlayer:" 14 bytes
    // in_traversal_part: (length) bytes
    // byte_a, byte_b, byte_c: 1 byte each
    int byte_a_idx = thread / (available_char_length_pow_2);
    int byte_b_idx = (thread % (available_char_length_pow_2)) / available_char_length;
    int byte_c_idx = thread % available_char_length;

    u8 byte_a = available_chars[byte_a_idx];
    u8 byte_b = available_chars[byte_b_idx];
    u8 byte_c = available_chars[byte_c_idx];

    // best result within this thread
    u8 local_best_in[player_name_max_length] = {0};
    u8 local_best_out[md5_block_size] = {0};

    for (unsigned char& i : local_best_in)
    {
        i = UINT8_MAX;
    }
    for (unsigned char& i : local_best_out)
    {
        i = UINT8_MAX;
    }

    // in_traversal_part is a base 63 integer as the index of available_chars
    u8 in_traversal_part[player_name_max_length] = { 0 };

    // iterate through all possible player names with (length + 3) characters
    for (int _ = 0; _ < pow(available_char_length, length); ++_)
    {
        // add 1 to in_traversal_part
        get_next_index(in_traversal_part, length);

        // assemble the MD5 input
        u8 in[player_name_prefix_length + player_name_max_length] = { 0 };
        for (int i = 0; i < player_name_prefix_length; ++i)
        {
            in[i] = player_name_prefix[i];
        }
        for (int i = length - 1; i >= 0; --i)
        {
            in[player_name_prefix_length + i] = available_chars[in_traversal_part[i]];
        }
        in[player_name_prefix_length + length] = byte_a;
        in[player_name_prefix_length + length + 1] = byte_b;
        in[player_name_prefix_length + length + 2] = byte_c;

        // calculate MD5 hash
        u32 inlen = player_name_prefix_length + length + 3;
        u8 out[md5_block_size];

        cuda_md5_ctx ctx;
        cuda_md5_init(&ctx);
        cuda_md5_update(&ctx, in, inlen);
        cuda_md5_final(&ctx, out);

        // compare with the best result within this thread
        // msvc does not support u128, so we have to compare hi and lo separately
        u64 current_out_hi, current_out_lo, local_best_out_hi, local_best_out_lo;

        convert_md5_to_u128(out, &current_out_hi, &current_out_lo);
        convert_md5_to_u128(cuda_outdata, &local_best_out_hi, &local_best_out_lo);

        if (current_out_hi < local_best_out_hi || (current_out_hi == local_best_out_hi && current_out_lo < local_best_out_lo))
        {
            for (int i = 0; i < player_name_max_length; ++i)
            {
                local_best_in[i] = in[player_name_prefix_length + i];
            }
            for (int i = 0; i < md5_block_size; ++i)
            {
                local_best_out[i] = out[i];
            }
        }
    }

    // write the best result within this thread to global memory
    for (int i = 0; i < player_name_max_length; ++i)
    {
        cuda_indata[thread * player_name_max_length + i] = local_best_in[i];
    }
    for (int i = 0; i < md5_block_size; ++i)
    {
        cuda_outdata[thread * md5_block_size + i] = local_best_out[i];
    }
}

int main()
{
    // test 1: n-batch md5 hash
    u8 in[21] = {
        'O', 'f', 'f', 'l', 'i', 'n', 'e', 'P', 'l', 'a', 'y', 'e', 'r', ':', 'C', 'a', 't', 'M', 'e', '0', 'w'
    };
    u8 out[md5_block_size];
    mcm_cuda_md5_hash_batch(in, 21, out, 1);
    for (const u8 i : out)
    {
        printf("%02x", i);
    }
    printf("\n");

    // test 2: 6 chars player name md5 hash
    u8* cuda_indata;
    u8* cuda_outdata;
    cudaMalloc(&cuda_indata, available_char_length_pow_3 * player_name_max_length);
    cudaMalloc(&cuda_outdata, available_char_length_pow_3 * md5_block_size);

    // 250112 threads (250047 used), 977 blocks
    int thread = 256;
    int block = (available_char_length_pow_3 + thread - 1) / thread;

    kernel_md5_hash_player_name << < block, thread >> > (3, cuda_indata, cuda_outdata);
    cudaDeviceSynchronize();

    auto indata = new u8[available_char_length_pow_3 * player_name_max_length];
    auto outdata = new u8[available_char_length_pow_3 * md5_block_size];
    cudaMemcpy(indata, cuda_indata, available_char_length_pow_3 * player_name_max_length, cudaMemcpyDeviceToHost);
    cudaMemcpy(outdata, cuda_outdata, available_char_length_pow_3 * md5_block_size, cudaMemcpyDeviceToHost);

    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("Error kernel_md5_hash_player_name: %s \n", cudaGetErrorString(error));
    }

    cudaFree(cuda_indata);
    cudaFree(cuda_outdata);

    // find the best player name in the results
    u8 best_in[player_name_max_length] = { 0 };
    u64 best_out_hi = ULLONG_MAX;
    u64 best_out_lo = ULLONG_MAX;

    for (int i = 0; i < available_char_length_pow_3; ++i)
    {
        u64 hi, lo;
        convert_md5_to_u128_cpu(outdata + i * md5_block_size, &hi, &lo);
        if (hi < best_out_hi || (hi == best_out_hi && lo < best_out_lo))
        {
            for (int j = 0; j < player_name_max_length; ++j)
            {
                best_in[j] = indata[i * player_name_max_length + j];
            }
            best_out_hi = hi;
            best_out_lo = lo;
        }
    }

    printf("Best player name: ");
    for (const u8 i : best_in)
    {
        printf("%c", i);
    }
    printf("\nMD5: ");
    printf("%016llx%016llx\n", best_out_hi, best_out_lo);

    return 0;
}
