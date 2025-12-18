#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>

#include "common.h"

__constant__ u8 player_name_prefix[] = {
    'O', 'f', 'f', 'l', 'i', 'n', 'e', 'P', 'l', 'a', 'y', 'e', 'r', ':'
};

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

// md5 constants and macros
// from https://github.com/B-Con/crypto-algorithms by Brad Conte
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))

#define F(x,y,z) (((x) & (y)) | (~(x) & (z)))
#define G(x,y,z) (((x) & (z)) | ((y) & ~(z)))
#define H(x,y,z) ((x) ^ (y) ^ (z))
#define I(x,y,z) ((y) ^ ((x) | ~(z)))

#define FF(a,b,c,d,m,s,t) { (a) += F(b,c,d) + (m) + (t); \
                            (a) = (b) + ROTLEFT(a,s); }
#define GG(a,b,c,d,m,s,t) { (a) += G(b,c,d) + (m) + (t); \
                            (a) = (b) + ROTLEFT(a,s); }
#define HH(a,b,c,d,m,s,t) { (a) += H(b,c,d) + (m) + (t); \
                            (a) = (b) + ROTLEFT(a,s); }
#define II(a,b,c,d,m,s,t) { (a) += I(b,c,d) + (m) + (t); \
                            (a) = (b) + ROTLEFT(a,s); }

__device__ __forceinline__ void md5_transform(u32 state[4], const u32 data[16])
{
    u32 a = state[0];
    u32 b = state[1];
    u32 c = state[2];
    u32 d = state[3];

    FF(a, b, c, d, data[0], 7, 0xd76aa478)
    FF(d, a, b, c, data[1], 12, 0xe8c7b756)
    FF(c, d, a, b, data[2], 17, 0x242070db)
    FF(b, c, d, a, data[3], 22, 0xc1bdceee)
    FF(a, b, c, d, data[4], 7, 0xf57c0faf)
    FF(d, a, b, c, data[5], 12, 0x4787c62a)
    FF(c, d, a, b, data[6], 17, 0xa8304613)
    FF(b, c, d, a, data[7], 22, 0xfd469501)
    FF(a, b, c, d, data[8], 7, 0x698098d8)
    FF(d, a, b, c, data[9], 12, 0x8b44f7af)
    FF(c, d, a, b, data[10], 17, 0xffff5bb1)
    FF(b, c, d, a, data[11], 22, 0x895cd7be)
    FF(a, b, c, d, data[12], 7, 0x6b901122)
    FF(d, a, b, c, data[13], 12, 0xfd987193)
    FF(c, d, a, b, data[14], 17, 0xa679438e)
    FF(b, c, d, a, data[15], 22, 0x49b40821)

    GG(a, b, c, d, data[1], 5, 0xf61e2562)
    GG(d, a, b, c, data[6], 9, 0xc040b340)
    GG(c, d, a, b, data[11], 14, 0x265e5a51)
    GG(b, c, d, a, data[0], 20, 0xe9b6c7aa)
    GG(a, b, c, d, data[5], 5, 0xd62f105d)
    GG(d, a, b, c, data[10], 9, 0x02441453)
    GG(c, d, a, b, data[15], 14, 0xd8a1e681)
    GG(b, c, d, a, data[4], 20, 0xe7d3fbc8)
    GG(a, b, c, d, data[9], 5, 0x21e1cde6)
    GG(d, a, b, c, data[14], 9, 0xc33707d6)
    GG(c, d, a, b, data[3], 14, 0xf4d50d87)
    GG(b, c, d, a, data[8], 20, 0x455a14ed)
    GG(a, b, c, d, data[13], 5, 0xa9e3e905)
    GG(d, a, b, c, data[2], 9, 0xfcefa3f8)
    GG(c, d, a, b, data[7], 14, 0x676f02d9)
    GG(b, c, d, a, data[12], 20, 0x8d2a4c8a)

    HH(a, b, c, d, data[5], 4, 0xfffa3942)
    HH(d, a, b, c, data[8], 11, 0x8771f681)
    HH(c, d, a, b, data[11], 16, 0x6d9d6122)
    HH(b, c, d, a, data[14], 23, 0xfde5380c)
    HH(a, b, c, d, data[1], 4, 0xa4beea44)
    HH(d, a, b, c, data[4], 11, 0x4bdecfa9)
    HH(c, d, a, b, data[7], 16, 0xf6bb4b60)
    HH(b, c, d, a, data[10], 23, 0xbebfbc70)
    HH(a, b, c, d, data[13], 4, 0x289b7ec6)
    HH(d, a, b, c, data[0], 11, 0xeaa127fa)
    HH(c, d, a, b, data[3], 16, 0xd4ef3085)
    HH(b, c, d, a, data[6], 23, 0x04881d05)
    HH(a, b, c, d, data[9], 4, 0xd9d4d039)
    HH(d, a, b, c, data[12], 11, 0xe6db99e5)
    HH(c, d, a, b, data[15], 16, 0x1fa27cf8)
    HH(b, c, d, a, data[2], 23, 0xc4ac5665)

    II(a, b, c, d, data[0], 6, 0xf4292244)
    II(d, a, b, c, data[7], 10, 0x432aff97)
    II(c, d, a, b, data[14], 15, 0xab9423a7)
    II(b, c, d, a, data[5], 21, 0xfc93a039)
    II(a, b, c, d, data[12], 6, 0x655b59c3)
    II(d, a, b, c, data[3], 10, 0x8f0ccc92)
    II(c, d, a, b, data[10], 15, 0xffeff47d)
    II(b, c, d, a, data[1], 21, 0x85845dd1)
    II(a, b, c, d, data[8], 6, 0x6fa87e4f)
    II(d, a, b, c, data[15], 10, 0xfe2ce6e0)
    II(c, d, a, b, data[6], 15, 0xa3014314)
    II(b, c, d, a, data[13], 21, 0x4e0811a1)
    II(a, b, c, d, data[4], 6, 0xf7537e82)

    // early stop: check if a matches the requirement for state[0] == 0
    // state[0] starts as 0x67452301. We need state[0] + a == 0.
    if (state[0] + a != 0) return;

    II(d, a, b, c, data[11], 10, 0xbd3af235)
    II(c, d, a, b, data[2], 15, 0x2ad7d2bb)
    II(b, c, d, a, data[9], 21, 0xeb86d391)

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
}

template <int Length>
__global__ void kernel_md5_hash_player_name_t()
{
    const u32 thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread >= available_char_length_pow_3)
    {
        return;
    }

    const int byte_a_idx = thread / available_char_length_pow_2;
    const int byte_b_idx = thread % available_char_length_pow_2 / available_char_length;
    const int byte_c_idx = thread % available_char_length;

    const u8 byte_a = available_chars[byte_a_idx];
    const u8 byte_b = available_chars[byte_b_idx];
    const u8 byte_c = available_chars[byte_c_idx];

    // in_traversal_part is a base 63 integer as the index of available_chars
    u8 in_traversal_part[Length > 0 ? Length : 1] = {0};

    // prepare MD5 block
    u32 block[16] = {0};

    // pre-fill constant parts
    // "OfflinePlayer:" -> 14 bytes
    // bytes 0-11 are fully constant, so we pre-fill block[0], block[1], block[2]
    // bytes 12-13 ('r', ':') are in block[3], which is mixed with variable data, so it's handled in the loop
    block[0] = 0x6c66664f; // 'Offl' (in little-endian, same for others)
    block[1] = 0x50656e69; // 'ineP'
    block[2] = 0x6579616c; // 'laye'

    // pre-fill length bits
    const u64 bitlen = (14 + Length + 3) * 8;
    block[14] = static_cast<u32>(bitlen);
    block[15] = static_cast<u32>(bitlen >> 32);

    // iterate through all possible player names with (length + 3) characters
    const int thread_max_iteration_count = pow(available_char_length, Length);

#pragma unroll
    for (int _ = 0; _ < thread_max_iteration_count; ++_)
    {
        // add 1 to in_traversal_part
        if (Length > 0)
        {
            int carry = 1;
#pragma unroll
            for (int i = Length - 1; i >= 0 && carry > 0; --i)
            {
                in_traversal_part[i] += carry;
                if (in_traversal_part[i] >= available_char_length)
                {
                    in_traversal_part[i] = 0;
                    carry = 1;
                }
                else
                {
                    carry = 0;
                }
            }
        }

        // construct md5 block
#pragma unroll
        for (int w = 3; w <= 7; ++w)
        {
            u32 word = 0;
#pragma unroll
            for (int b = 0; b < 4; ++b)
            {
                const int p = w * 4 + b;
                u8 val;
                if (p < 14) val = player_name_prefix[p]; // prefix; filling "r:" here since "OfflinePlaye" is filled above
                else if (p < 14 + Length) val = available_chars[in_traversal_part[p - 14]]; // "variable" part of playername
                else if (p == 14 + Length) val = byte_a;
                else if (p == 14 + Length + 1) val = byte_b;
                else if (p == 14 + Length + 2) val = byte_c;
                else if (p == 14 + Length + 3) val = 0x80; // 0x80 padding
                else val = 0; // zero padding

                word |= static_cast<u32>(val) << (b * 8);
            }
            block[w] = word;
        }

        // md5 transform
        u32 state[4] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476}; // md5 initial state
        md5_transform(state, block);

        // check for 8 leading zeros (32 bits)
        if (state[0] == 0)
        {
            const u64 s0 = state[0];
            const u64 s1 = state[1];
            const u64 s2 = state[2];
            const u64 s3 = state[3];

            // reconstruct hi/lo for UUID generation
            const u64 r0_val = (s0 & 0xFF) << 24 | (s0 & 0xFF00) << 8 | (s0 & 0xFF0000) >> 8 | s0 >> 24 & 0xFF;
            const u64 r1_val = (s1 & 0xFF) << 24 | (s1 & 0xFF00) << 8 | (s1 & 0xFF0000) >> 8 | s1 >> 24 & 0xFF;
            const u64 current_out_hi = r0_val << 32 | r1_val;

            const u64 r2_val = (s2 & 0xFF) << 24 | (s2 & 0xFF00) << 8 | (s2 & 0xFF0000) >> 8 | s2 >> 24 & 0xFF;
            const u64 r3_val = (s3 & 0xFF) << 24 | (s3 & 0xFF00) << 8 | (s3 & 0xFF0000) >> 8 | s3 >> 24 & 0xFF;
            const u64 current_out_lo = r2_val << 32 | r3_val;

            const u64 final_hi = current_out_hi & 0xFFFFFFFFFFFF0FFFULL | 0x0000000000003000ULL;
            const u64 final_lo = current_out_lo & 0x3FFFFFFFFFFFFFFFULL | 0x8000000000000000ULL;

            char name_buf[player_name_max_length + 1];
            for (int k = 0; k < Length; ++k) name_buf[k] = available_chars[in_traversal_part[k]];
            name_buf[Length] = byte_a;
            name_buf[Length + 1] = byte_b;
            name_buf[Length + 2] = byte_c;
            name_buf[Length + 3] = '\0';

            printf("%s,%08x-%04x-%04x-%04x-%04x%08x\n",
                   name_buf,
                   static_cast<u32>(final_hi >> 32),
                   static_cast<u32>(final_hi >> 16 & 0xFFFF),
                   static_cast<u32>(final_hi & 0xFFFF),
                   static_cast<u32>(final_lo >> 48 & 0xFFFF),
                   static_cast<u32>(final_lo >> 32 & 0xFFFF),
                   static_cast<u32>(final_lo & 0xFFFFFFFF)
            );
        }
    }
}

int main()
{
    // 250112 threads (250047 used), 977 blocks
    constexpr int thread = 256;
    constexpr int block = (available_char_length_pow_3 + thread - 1) / thread;

    for (int i = 0; i <= player_name_max_length - 3; ++i)
    {
        fprintf(stderr, "Searching %d-character player names...\n", i + 3);

        // measure time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // launch kernel
        switch (i)
        {
        case 0: kernel_md5_hash_player_name_t<0><<<block, thread>>>();
            break;
        case 1: kernel_md5_hash_player_name_t<1><<<block, thread>>>();
            break;
        case 2: kernel_md5_hash_player_name_t<2><<<block, thread>>>();
            break;
        case 3: kernel_md5_hash_player_name_t<3><<<block, thread>>>();
            break;
        case 4: kernel_md5_hash_player_name_t<4><<<block, thread>>>();
            break;
        case 5: kernel_md5_hash_player_name_t<5><<<block, thread>>>();
            break;
        case 6: kernel_md5_hash_player_name_t<6><<<block, thread>>>();
            break;
        case 7: kernel_md5_hash_player_name_t<7><<<block, thread>>>();
            break;
        case 8: kernel_md5_hash_player_name_t<8><<<block, thread>>>();
            break;
        case 9: kernel_md5_hash_player_name_t<9><<<block, thread>>>();
            break;
        case 10: kernel_md5_hash_player_name_t<10><<<block, thread>>>();
            break;
        case 11: kernel_md5_hash_player_name_t<11><<<block, thread>>>();
            break;
        case 12: kernel_md5_hash_player_name_t<12><<<block, thread>>>();
            break;
        case 13: kernel_md5_hash_player_name_t<13><<<block, thread>>>();
            break;
        default: ;
        }

        cudaDeviceSynchronize();

        // measure time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            fprintf(stderr, "Error kernel_md5_hash_player_name: %s \n", cudaGetErrorString(error));
        }

        fprintf(stderr, "Time elapsed: %.3f s\n\n", milliseconds / 1000.0f);
    }

    fprintf(stderr, "Press any key to exit...");
    (void)getchar();

    return 0;
}
