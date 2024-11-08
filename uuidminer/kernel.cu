#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>

#include "md5.cuh"

int main()
{
	u8 in[21] = {
		'O', 'f', 'f', 'l', 'i', 'n', 'e', 'P', 'l', 'a', 'y', 'e', 'r', ':', 'C', 'a', 't', 'M', 'e', '0', 'w'
	};
	u8 out[MD5_BLOCK_SIZE];
	mcm_cuda_md5_hash_batch(in, 21, out, 1);
	for (const u8 i : out)
	{
		printf("%02x ", i);
	}

	return 0;
}
