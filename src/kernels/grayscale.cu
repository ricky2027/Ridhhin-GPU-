#include "kernels.h"

#include <cuda_runtime.h>

namespace {

// ITU-R BT.601 luma coefficients.
constexpr float kLumaRed = 0.299f;
constexpr float kLumaGreen = 0.587f;
constexpr float kLumaBlue = 0.114f;

constexpr int kBlockSize = 256;

// Converts one pixel from interleaved RGB(A) to a single grayscale byte.
// Each thread handles exactly one output pixel. 1D grid.
__global__ void GrayscaleKernel(const unsigned char* input,
                                unsigned char* output, int num_pixels,
                                int channels) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pixels) {
    return;
  }

  const int base = idx * channels;
  const float r = static_cast<float>(input[base + 0]);
  const float g = static_cast<float>(input[base + 1]);
  const float b = static_cast<float>(input[base + 2]);

  const float luma = kLumaRed * r + kLumaGreen * g + kLumaBlue * b;
  output[idx] = static_cast<unsigned char>(luma + 0.5f);
}

}  // namespace

void LaunchGrayscale(const unsigned char* d_input, unsigned char* d_output,
                     int width, int height, int channels) {
  const int num_pixels = width * height;
  const int grid_size = (num_pixels + kBlockSize - 1) / kBlockSize;
  GrayscaleKernel<<<grid_size, kBlockSize>>>(d_input, d_output, num_pixels,
                                             channels);
}
