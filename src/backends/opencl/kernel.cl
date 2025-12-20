__constant uchar player_name_prefix[] = {'O', 'f', 'f', 'l', 'i', 'n', 'e', 'P', 'l', 'a', 'y', 'e', 'r', ':'};

__constant uchar available_chars[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
                                      'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                                      'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                                      'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_'};

#define available_char_length 63
#define available_char_length_pow_2 (63 * 63)
#define available_char_length_pow_3 (63 * 63 * 63)
#define player_name_max_length 16

#define ROTLEFT(a, b) (((a) << (b)) | ((a) >> (32 - (b))))

#define F(x, y, z) (((x) & (y)) | (~(x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & ~(z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | ~(z)))

#define FF(a, b, c, d, m, s, t)                                                                                        \
    {                                                                                                                  \
        (a) += F(b, c, d) + (m) + (t);                                                                                 \
        (a) = (b) + ROTLEFT(a, s);                                                                                     \
    }
#define GG(a, b, c, d, m, s, t)                                                                                        \
    {                                                                                                                  \
        (a) += G(b, c, d) + (m) + (t);                                                                                 \
        (a) = (b) + ROTLEFT(a, s);                                                                                     \
    }
#define HH(a, b, c, d, m, s, t)                                                                                        \
    {                                                                                                                  \
        (a) += H(b, c, d) + (m) + (t);                                                                                 \
        (a) = (b) + ROTLEFT(a, s);                                                                                     \
    }
#define II(a, b, c, d, m, s, t)                                                                                        \
    {                                                                                                                  \
        (a) += I(b, c, d) + (m) + (t);                                                                                 \
        (a) = (b) + ROTLEFT(a, s);                                                                                     \
    }

void md5_transform(uint state[4], const uint data[16], const uint target_state0)
{
    uint a = state[0];
    uint b = state[1];
    uint c = state[2];
    uint d = state[3];

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

    // early stop
    if (state[0] + a != target_state0)
        return;

    II(d, a, b, c, data[11], 10, 0xbd3af235)
    II(c, d, a, b, data[2], 15, 0x2ad7d2bb)
    II(b, c, d, a, data[9], 21, 0xeb86d391)

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
}

#define DEFINE_MD5_KERNEL(LEN)                                                                                         \
    __kernel void md5_hash_player_name_##LEN(const unsigned int offset, const unsigned int target_state0)              \
    {                                                                                                                  \
        const uint thread_id = get_global_id(0) + offset;                                                              \
        if (thread_id >= available_char_length_pow_3)                                                                  \
        {                                                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        const int byte_a_idx = thread_id / available_char_length_pow_2;                                                \
        const int byte_b_idx = thread_id % available_char_length_pow_2 / available_char_length;                        \
        const int byte_c_idx = thread_id % available_char_length;                                                      \
                                                                                                                       \
        const uchar byte_a = available_chars[byte_a_idx];                                                              \
        const uchar byte_b = available_chars[byte_b_idx];                                                              \
        const uchar byte_c = available_chars[byte_c_idx];                                                              \
                                                                                                                       \
        uchar in_traversal_part[16] = {0};                                                                             \
                                                                                                                       \
        uint block[16] = {0};                                                                                          \
                                                                                                                       \
        block[0] = 0x6c66664f;                                                                                         \
        block[1] = 0x50656e69;                                                                                         \
        block[2] = 0x6579616c;                                                                                         \
                                                                                                                       \
        const ulong bitlen = (14 + LEN + 3) * 8;                                                                       \
        block[14] = (uint)(bitlen);                                                                                    \
        block[15] = (uint)(bitlen >> 32);                                                                              \
                                                                                                                       \
        int thread_max_iteration_count = 1;                                                                            \
        for (int _ = 0; _ < LEN; ++_)                                                                                  \
            thread_max_iteration_count *= available_char_length;                                                       \
                                                                                                                       \
        for (int _ = 0; _ < thread_max_iteration_count; ++_)                                                           \
        {                                                                                                              \
            if (LEN > 0)                                                                                               \
            {                                                                                                          \
                int carry = 1;                                                                                         \
                for (int i = LEN - 1; i >= 0 && carry > 0; --i)                                                        \
                {                                                                                                      \
                    in_traversal_part[i] += carry;                                                                     \
                    if (in_traversal_part[i] >= available_char_length)                                                 \
                    {                                                                                                  \
                        in_traversal_part[i] = 0;                                                                      \
                        carry = 1;                                                                                     \
                    }                                                                                                  \
                    else                                                                                               \
                    {                                                                                                  \
                        carry = 0;                                                                                     \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
                                                                                                                       \
            for (int w = 3; w <= 7; ++w)                                                                               \
            {                                                                                                          \
                uint word = 0;                                                                                         \
                for (int b = 0; b < 4; ++b)                                                                            \
                {                                                                                                      \
                    const int p = w * 4 + b;                                                                           \
                    uchar val;                                                                                         \
                    if (p < 14)                                                                                        \
                        val = player_name_prefix[p];                                                                   \
                    else if (p < 14 + LEN)                                                                             \
                        val = available_chars[in_traversal_part[p - 14]];                                              \
                    else if (p == 14 + LEN)                                                                            \
                        val = byte_a;                                                                                  \
                    else if (p == 14 + LEN + 1)                                                                        \
                        val = byte_b;                                                                                  \
                    else if (p == 14 + LEN + 2)                                                                        \
                        val = byte_c;                                                                                  \
                    else if (p == 14 + LEN + 3)                                                                        \
                        val = 0x80;                                                                                    \
                    else                                                                                               \
                        val = 0;                                                                                       \
                                                                                                                       \
                    word |= (uint)(val) << (b * 8);                                                                    \
                }                                                                                                      \
                block[w] = word;                                                                                       \
            }                                                                                                          \
                                                                                                                       \
            uint state[4] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476};                                          \
            md5_transform(state, block, target_state0);                                                                \
                                                                                                                       \
            if (state[0] == target_state0)                                                                             \
            {                                                                                                          \
                const ulong s0 = state[0];                                                                             \
                const ulong s1 = state[1];                                                                             \
                const ulong s2 = state[2];                                                                             \
                const ulong s3 = state[3];                                                                             \
                                                                                                                       \
                const ulong r0_val = (s0 & 0xFF) << 24 | (s0 & 0xFF00) << 8 | (s0 & 0xFF0000) >> 8 | s0 >> 24 & 0xFF;  \
                const ulong r1_val = (s1 & 0xFF) << 24 | (s1 & 0xFF00) << 8 | (s1 & 0xFF0000) >> 8 | s1 >> 24 & 0xFF;  \
                const ulong current_out_hi = r0_val << 32 | r1_val;                                                    \
                                                                                                                       \
                const ulong r2_val = (s2 & 0xFF) << 24 | (s2 & 0xFF00) << 8 | (s2 & 0xFF0000) >> 8 | s2 >> 24 & 0xFF;  \
                const ulong r3_val = (s3 & 0xFF) << 24 | (s3 & 0xFF00) << 8 | (s3 & 0xFF0000) >> 8 | s3 >> 24 & 0xFF;  \
                const ulong current_out_lo = r2_val << 32 | r3_val;                                                    \
                                                                                                                       \
                const ulong final_hi = current_out_hi & 0xFFFFFFFFFFFF0FFFULL | 0x0000000000003000ULL;                 \
                const ulong final_lo = current_out_lo & 0x3FFFFFFFFFFFFFFFULL | 0x8000000000000000ULL;                 \
                                                                                                                       \
                char name_buf[player_name_max_length + 1];                                                             \
                for (int k = 0; k < LEN; ++k)                                                                          \
                    name_buf[k] = available_chars[in_traversal_part[k]];                                               \
                name_buf[LEN] = byte_a;                                                                                \
                name_buf[LEN + 1] = byte_b;                                                                            \
                name_buf[LEN + 2] = byte_c;                                                                            \
                name_buf[LEN + 3] = '\0';                                                                              \
                                                                                                                       \
                printf("%s,%08x-%04x-%04x-%04x-%04x%08x\n", name_buf, (uint)(final_hi >> 32),                          \
                       (uint)(final_hi >> 16 & 0xFFFF), (uint)(final_hi & 0xFFFF), (uint)(final_lo >> 48 & 0xFFFF),    \
                       (uint)(final_lo >> 32 & 0xFFFF), (uint)(final_lo & 0xFFFFFFFF));                                \
            }                                                                                                          \
        }                                                                                                              \
    }

DEFINE_MD5_KERNEL(0)
DEFINE_MD5_KERNEL(1)
DEFINE_MD5_KERNEL(2)
DEFINE_MD5_KERNEL(3)
DEFINE_MD5_KERNEL(4)
DEFINE_MD5_KERNEL(5)
DEFINE_MD5_KERNEL(6)
DEFINE_MD5_KERNEL(7)
DEFINE_MD5_KERNEL(8)
DEFINE_MD5_KERNEL(9)
DEFINE_MD5_KERNEL(10)
DEFINE_MD5_KERNEL(11)
DEFINE_MD5_KERNEL(12)
DEFINE_MD5_KERNEL(13)
