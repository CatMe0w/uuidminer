#include <metal_stdlib>
using namespace metal;

constant uchar player_name_prefix[] = {
    'O', 'f', 'f', 'l', 'i', 'n', 'e', 'P', 'l', 'a', 'y', 'e', 'r', ':'
};

constant uchar available_chars[] = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
    'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z',
    '_'
};

enum : uint {
    kAvailableCharLength = 63u,
    kAvailableCharLengthPow2 = 63u * 63u,
    kAvailableCharLengthPow3 = 63u * 63u * 63u,
    kPlayerNameMaxLength = 16u,
    kMaxResults = 1024u,
};

struct Params {
    uint start;
    uint end;
    uint target_state0;
    uint _pad;
    ulong iter_start;
    ulong iter_end;
};

struct Result {
    uint name_len;
    uchar name[kPlayerNameMaxLength];
    uint _pad;
    ulong final_hi;
    ulong final_lo;
};

static inline uint ROTLEFT(uint a, uint b) { return (a << b) | (a >> (32u - b)); }

static inline uint F(uint x, uint y, uint z) { return (x & y) | (~x & z); }
static inline uint G(uint x, uint y, uint z) { return (x & z) | (y & ~z); }
static inline uint H(uint x, uint y, uint z) { return x ^ y ^ z; }
static inline uint I(uint x, uint y, uint z) { return y ^ (x | ~z); }

#define FF(a, b, c, d, m, s, t)              \
    do {                                     \
        (a) += F((b), (c), (d)) + (m) + (t); \
        (a) = (b) + ROTLEFT((a), (s));       \
    } while (0)
#define GG(a, b, c, d, m, s, t)              \
    do {                                     \
        (a) += G((b), (c), (d)) + (m) + (t); \
        (a) = (b) + ROTLEFT((a), (s));       \
    } while (0)
#define HH(a, b, c, d, m, s, t)              \
    do {                                     \
        (a) += H((b), (c), (d)) + (m) + (t); \
        (a) = (b) + ROTLEFT((a), (s));       \
    } while (0)
#define II(a, b, c, d, m, s, t)              \
    do {                                     \
        (a) += I((b), (c), (d)) + (m) + (t); \
        (a) = (b) + ROTLEFT((a), (s));       \
    } while (0)

static inline void md5_transform(thread uint state[4], thread const uint data[16], uint target_state0)
{
    uint a = state[0];
    uint b = state[1];
    uint c = state[2];
    uint d = state[3];

    FF(a, b, c, d, data[0], 7, 0xd76aa478);
    FF(d, a, b, c, data[1], 12, 0xe8c7b756);
    FF(c, d, a, b, data[2], 17, 0x242070db);
    FF(b, c, d, a, data[3], 22, 0xc1bdceee);
    FF(a, b, c, d, data[4], 7, 0xf57c0faf);
    FF(d, a, b, c, data[5], 12, 0x4787c62a);
    FF(c, d, a, b, data[6], 17, 0xa8304613);
    FF(b, c, d, a, data[7], 22, 0xfd469501);
    FF(a, b, c, d, data[8], 7, 0x698098d8);
    FF(d, a, b, c, data[9], 12, 0x8b44f7af);
    FF(c, d, a, b, data[10], 17, 0xffff5bb1);
    FF(b, c, d, a, data[11], 22, 0x895cd7be);
    FF(a, b, c, d, data[12], 7, 0x6b901122);
    FF(d, a, b, c, data[13], 12, 0xfd987193);
    FF(c, d, a, b, data[14], 17, 0xa679438e);
    FF(b, c, d, a, data[15], 22, 0x49b40821);

    GG(a, b, c, d, data[1], 5, 0xf61e2562);
    GG(d, a, b, c, data[6], 9, 0xc040b340);
    GG(c, d, a, b, data[11], 14, 0x265e5a51);
    GG(b, c, d, a, data[0], 20, 0xe9b6c7aa);
    GG(a, b, c, d, data[5], 5, 0xd62f105d);
    GG(d, a, b, c, data[10], 9, 0x02441453);
    GG(c, d, a, b, data[15], 14, 0xd8a1e681);
    GG(b, c, d, a, data[4], 20, 0xe7d3fbc8);
    GG(a, b, c, d, data[9], 5, 0x21e1cde6);
    GG(d, a, b, c, data[14], 9, 0xc33707d6);
    GG(c, d, a, b, data[3], 14, 0xf4d50d87);
    GG(b, c, d, a, data[8], 20, 0x455a14ed);
    GG(a, b, c, d, data[13], 5, 0xa9e3e905);
    GG(d, a, b, c, data[2], 9, 0xfcefa3f8);
    GG(c, d, a, b, data[7], 14, 0x676f02d9);
    GG(b, c, d, a, data[12], 20, 0x8d2a4c8a);

    HH(a, b, c, d, data[5], 4, 0xfffa3942);
    HH(d, a, b, c, data[8], 11, 0x8771f681);
    HH(c, d, a, b, data[11], 16, 0x6d9d6122);
    HH(b, c, d, a, data[14], 23, 0xfde5380c);
    HH(a, b, c, d, data[1], 4, 0xa4beea44);
    HH(d, a, b, c, data[4], 11, 0x4bdecfa9);
    HH(c, d, a, b, data[7], 16, 0xf6bb4b60);
    HH(b, c, d, a, data[10], 23, 0xbebfbc70);
    HH(a, b, c, d, data[13], 4, 0x289b7ec6);
    HH(d, a, b, c, data[0], 11, 0xeaa127fa);
    HH(c, d, a, b, data[3], 16, 0xd4ef3085);
    HH(b, c, d, a, data[6], 23, 0x04881d05);
    HH(a, b, c, d, data[9], 4, 0xd9d4d039);
    HH(d, a, b, c, data[12], 11, 0xe6db99e5);
    HH(c, d, a, b, data[15], 16, 0x1fa27cf8);
    HH(b, c, d, a, data[2], 23, 0xc4ac5665);

    II(a, b, c, d, data[0], 6, 0xf4292244);
    II(d, a, b, c, data[7], 10, 0x432aff97);
    II(c, d, a, b, data[14], 15, 0xab9423a7);
    II(b, c, d, a, data[5], 21, 0xfc93a039);
    II(a, b, c, d, data[12], 6, 0x655b59c3);
    II(d, a, b, c, data[3], 10, 0x8f0ccc92);
    II(c, d, a, b, data[10], 15, 0xffeff47d);
    II(b, c, d, a, data[1], 21, 0x85845dd1);
    II(a, b, c, d, data[8], 6, 0x6fa87e4f);
    II(d, a, b, c, data[15], 10, 0xfe2ce6e0);
    II(c, d, a, b, data[6], 15, 0xa3014314);
    II(b, c, d, a, data[13], 21, 0x4e0811a1);
    II(a, b, c, d, data[4], 6, 0xf7537e82);

    // early stop
    if (state[0] + a != target_state0)
        return;

    II(d, a, b, c, data[11], 10, 0xbd3af235);
    II(c, d, a, b, data[2], 15, 0x2ad7d2bb);
    II(b, c, d, a, data[9], 21, 0xeb86d391);

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
}

template <ushort Length>
static inline void store_match(device atomic_uint* match_count, device Result* results, uint idx,
    thread const ushort traversal[Length > 0 ? Length : 1],
    uchar byte_a, uchar byte_b, uchar byte_c,
    thread const uint state[4])
{
    // only store up to a fixed maximum to keep host-side printing bounded
    if (idx >= kMaxResults)
        return;

    // reconstruct hi/lo for UUID generation (same as CUDA)
    const ulong s0 = (ulong)state[0];
    const ulong s1 = (ulong)state[1];
    const ulong s2 = (ulong)state[2];
    const ulong s3 = (ulong)state[3];

    const ulong r0_val = ((s0 & 0xFFul) << 24) | ((s0 & 0xFF00ul) << 8) | ((s0 & 0xFF0000ul) >> 8) | ((s0 >> 24) & 0xFFul);
    const ulong r1_val = ((s1 & 0xFFul) << 24) | ((s1 & 0xFF00ul) << 8) | ((s1 & 0xFF0000ul) >> 8) | ((s1 >> 24) & 0xFFul);
    const ulong current_out_hi = (r0_val << 32) | r1_val;

    const ulong r2_val = ((s2 & 0xFFul) << 24) | ((s2 & 0xFF00ul) << 8) | ((s2 & 0xFF0000ul) >> 8) | ((s2 >> 24) & 0xFFul);
    const ulong r3_val = ((s3 & 0xFFul) << 24) | ((s3 & 0xFF00ul) << 8) | ((s3 & 0xFF0000ul) >> 8) | ((s3 >> 24) & 0xFFul);
    const ulong current_out_lo = (r2_val << 32) | r3_val;

    const ulong final_hi = (current_out_hi & 0xFFFFFFFFFFFF0FFFull) | 0x0000000000003000ull;
    const ulong final_lo = (current_out_lo & 0x3FFFFFFFFFFFFFFFull) | 0x8000000000000000ull;

    Result r;
    r.name_len = (uint)(Length + 3);
    // fill prefix part of name
    for (ushort k = 0; k < Length; ++k)
        r.name[k] = available_chars[traversal[k]];
    r.name[Length + 0] = byte_a;
    r.name[Length + 1] = byte_b;
    r.name[Length + 2] = byte_c;
    // zero the rest for safety
    for (ushort k = (ushort)(Length + 3); k < (ushort)kPlayerNameMaxLength; ++k)
        r.name[k] = 0;
    r._pad = 0;
    r.final_hi = final_hi;
    r.final_lo = final_lo;

    results[idx] = r;
}

template <ushort Length>
static inline void md5_hash_player_name_impl(constant Params& p,
    device atomic_uint* match_count,
    device Result* results,
    uint tid)
{
    const uint thread_id = tid + p.start;
    if (thread_id >= p.end)
        return;

    const uint byte_a_idx = thread_id / kAvailableCharLengthPow2;
    const uint byte_b_idx = (thread_id % kAvailableCharLengthPow2) / kAvailableCharLength;
    const uint byte_c_idx = thread_id % kAvailableCharLength;

    const uchar byte_a = available_chars[byte_a_idx];
    const uchar byte_b = available_chars[byte_b_idx];
    const uchar byte_c = available_chars[byte_c_idx];

    // traversal is a base-63 integer (stored as indices in available_chars)
    ushort traversal[Length > 0 ? Length : 1];
    if constexpr (Length > 0) {
        ulong tmp = p.iter_start;
        for (int i = (int)Length - 1; i >= 0; --i) {
            traversal[i] = (ushort)(tmp % kAvailableCharLength);
            tmp /= kAvailableCharLength;
        }
    }

    // prepare MD5 block
    uint block[16] = { 0 };

    // pre-fill constant parts
    block[0] = 0x6c66664f; // 'Offl'
    block[1] = 0x50656e69; // 'ineP'
    block[2] = 0x6579616c; // 'laye'

    // pre-fill length bits
    const ulong bitlen = (ulong)((14u + (uint)Length + 3u) * 8u);
    block[14] = (uint)(bitlen & 0xFFFFFFFFul);
    block[15] = (uint)((bitlen >> 32) & 0xFFFFFFFFul);

    ulong iter_start = p.iter_start;
    ulong iter_end = p.iter_end;

    if constexpr (Length == 0) {
        // no traversal dimension for 3-character names
        iter_start = 0;
        iter_end = 1;
    }

    for (ulong iter = iter_start; iter < iter_end; ++iter) {

        // construct block
        for (uint w = 3; w <= 7; ++w) {
            uint word = 0;
            for (uint b = 0; b < 4; ++b) {
                const uint pos = w * 4u + b;
                uchar val;
                if (pos < 14u) {
                    val = player_name_prefix[pos];
                } else if (pos < 14u + (uint)Length) {
                    val = available_chars[traversal[pos - 14u]];
                } else if (pos == 14u + (uint)Length) {
                    val = byte_a;
                } else if (pos == 14u + (uint)Length + 1u) {
                    val = byte_b;
                } else if (pos == 14u + (uint)Length + 2u) {
                    val = byte_c;
                } else if (pos == 14u + (uint)Length + 3u) {
                    val = 0x80;
                } else {
                    val = 0;
                }
                word |= (uint)val << (b * 8u);
            }
            block[w] = word;
        }

        // md5 transform
        uint state[4] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u };
        md5_transform(state, block, p.target_state0);

        if (state[0] == p.target_state0) {
            const uint out_idx = atomic_fetch_add_explicit(match_count, 1u, memory_order_relaxed);
            store_match<Length>(match_count, results, out_idx, traversal, byte_a, byte_b, byte_c, state);
        }

        if constexpr (Length > 0) {
            // increment base-63 traversal for next iteration
            int carry = 1;
            for (int i = (int)Length - 1; i >= 0 && carry > 0; --i) {
                ushort v = (ushort)(traversal[i] + carry);
                if (v >= (ushort)kAvailableCharLength) {
                    traversal[i] = 0;
                    carry = 1;
                } else {
                    traversal[i] = v;
                    carry = 0;
                }
            }
        }
    }
}

#define DEFINE_MD5_KERNEL(L)                                               \
    kernel void md5_hash_player_name_##L(constant Params& p [[buffer(0)]], \
        device atomic_uint* match_count [[buffer(1)]],                     \
        device Result* results [[buffer(2)]],                              \
        uint tid [[thread_position_in_grid]])                              \
    {                                                                      \
        md5_hash_player_name_impl<L>(p, match_count, results, tid);        \
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
