#include "kernels.h"

#include <cuda_runtime.h>

namespace {

constexpr int kRadius = 2;         // 5x5 kernel -> radius 2.
constexpr int kKernelDim = 5;
constexpr int kBlockDim = 16;      // 16x16 thread blocks.
constexpr float kWeightSum = 256.0f;

// Standard 1/256 weighted 5x5 Gaussian kernel (unnormalized integer weights;
// the sum of all entries is 256). Stored in constant memory for broadcast
// access across the thread block.
__constant__ float kGaussianWeights[kKernelDim * kKernelDim] = {
    1.0f,  4.0f,  6.0f,  4.0f,  1.0f,
    4.0f, 16.0f, 24.0f, 16.0f,  4.0f,
    6.0f, 24.0f, 36.0f, 24.0f,  6.0f,
    4.0f, 16.0f, 24.0f, 16.0f,  4.0f,
    1.0f,  4.0f,  6.0f,  4.0f,  1.0f};

__device__ int Clamp(int value, int low, int high) {
  return value < low ? low : (value > high ? high : value);
}

// Applies the 5x5 Gaussian blur to one pixel for every channel. Border
// coordinates are clamped to the edge of the image. 2D grid.
__global__ void GaussianBlurKernel(const unsigned char* input,
                                   unsigned char* output, int width,
                                   int height, int channels) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  for (int c = 0; c < channels; ++c) {
    float sum = 0.0f;
    for (int ky = -kRadius; ky <= kRadius; ++ky) {
      const int sy = Clamp(y + ky, 0, height - 1);
      for (int kx = -kRadius; kx <= kRadius; ++kx) {
        const int sx = Clamp(x + kx, 0, width - 1);
        const float weight =
            kGaussianWeights[(ky + kRadius) * kKernelDim + (kx + kRadius)];
        sum += weight * static_cast<float>(input[(sy * width + sx) * channels + c]);
      }
    }
    output[(y * width + x) * channels + c] =
        static_cast<unsigned char>(sum / kWeightSum + 0.5f);
  }
}

}  // namespace

void LaunchGaussianBlur(const unsigned char* d_input, unsigned char* d_output,
                        int width, int height, int channels) {
  const dim3 block(kBlockDim, kBlockDim);
  const dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
  GaussianBlurKernel<<<grid, block>>>(d_input, d_output, width, height,
                                      channels);
}
