#include "kernels.h"

#include <cuda_runtime.h>

#include <cmath>

namespace {

constexpr int kBlockDim = 16;
constexpr int kMaxByte = 255;

// Sobel convolution kernels in constant memory.
__constant__ float kSobelGx[9] = {
    -1.0f, 0.0f, 1.0f,
    -2.0f, 0.0f, 2.0f,
    -1.0f, 0.0f, 1.0f};

__constant__ float kSobelGy[9] = {
    -1.0f, -2.0f, -1.0f,
     0.0f,  0.0f,  0.0f,
     1.0f,  2.0f,  1.0f};

__device__ int Clamp(int value, int low, int high) {
  return value < low ? low : (value > high ? high : value);
}

// Computes the Sobel gradient magnitude for one pixel of a grayscale image.
// Border samples are clamped to the edge. 2D grid.
__global__ void SobelEdgeKernel(const unsigned char* input,
                                unsigned char* output, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  float gx = 0.0f;
  float gy = 0.0f;
  for (int ky = -1; ky <= 1; ++ky) {
    const int sy = Clamp(y + ky, 0, height - 1);
    for (int kx = -1; kx <= 1; ++kx) {
      const int sx = Clamp(x + kx, 0, width - 1);
      const int tap = (ky + 1) * 3 + (kx + 1);
      const float value = static_cast<float>(input[sy * width + sx]);
      gx += kSobelGx[tap] * value;
      gy += kSobelGy[tap] * value;
    }
  }

  const int magnitude = static_cast<int>(sqrtf(gx * gx + gy * gy) + 0.5f);
  output[y * width + x] =
      static_cast<unsigned char>(Clamp(magnitude, 0, kMaxByte));
}

}  // namespace

void LaunchSobelEdge(const unsigned char* d_input, unsigned char* d_output,
                     int width, int height) {
  const dim3 block(kBlockDim, kBlockDim);
  const dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
  SobelEdgeKernel<<<grid, block>>>(d_input, d_output, width, height);
}
