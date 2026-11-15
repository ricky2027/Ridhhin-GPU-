#include "kernels.h"

#include <cuda_runtime.h>

namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxByte = 255;

// Inverts every byte of the image: output[i] = 255 - input[i]. Because it
// operates per byte, it covers all channels uniformly. 1D grid.
__global__ void NegativeKernel(const unsigned char* input,
                               unsigned char* output, int num_bytes) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_bytes) {
    return;
  }
  output[idx] = static_cast<unsigned char>(kMaxByte - input[idx]);
}

}  // namespace

void LaunchNegative(const unsigned char* d_input, unsigned char* d_output,
                    int num_bytes) {
  const int grid_size = (num_bytes + kBlockSize - 1) / kBlockSize;
  NegativeKernel<<<grid_size, kBlockSize>>>(d_input, d_output, num_bytes);
}
