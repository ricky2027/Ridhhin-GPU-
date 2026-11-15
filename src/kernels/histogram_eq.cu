#include "kernels.h"

#include <cuda_runtime.h>

namespace {

constexpr int kNumBins = 256;
constexpr int kBlockSize = 256;

// Builds the 256-bin intensity histogram of a grayscale image. Each thread
// processes one pixel and bumps the corresponding bin with atomicAdd. 1D grid.
__global__ void HistogramKernel(const unsigned char* input, int num_pixels,
                                unsigned int* histogram) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pixels) {
    return;
  }
  atomicAdd(&histogram[input[idx]], 1u);
}

// Remaps each pixel through the equalization lookup table. 1D grid.
__global__ void ApplyLutKernel(const unsigned char* input,
                               unsigned char* output, int num_pixels,
                               const unsigned char* lut) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pixels) {
    return;
  }
  output[idx] = lut[input[idx]];
}

}  // namespace

void LaunchHistogramEq(const unsigned char* d_input, unsigned char* d_output,
                       int width, int height) {
  const int num_pixels = width * height;
  const int grid_size = (num_pixels + kBlockSize - 1) / kBlockSize;

  // Step (a): compute the histogram on the GPU.
  unsigned int* d_histogram = nullptr;
  cudaMalloc(&d_histogram, kNumBins * sizeof(unsigned int));
  cudaMemset(d_histogram, 0, kNumBins * sizeof(unsigned int));
  HistogramKernel<<<grid_size, kBlockSize>>>(d_input, num_pixels, d_histogram);

  unsigned int histogram[kNumBins];
  cudaMemcpy(histogram, d_histogram, kNumBins * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

  // Step (b): compute the CDF on the CPU (only 256 values) and build the
  // normalized lookup table. The first non-zero CDF entry is used as the
  // minimum so the output stretches across the full [0, 255] range.
  unsigned int cdf[kNumBins];
  unsigned int running = 0;
  unsigned int cdf_min = 0;
  bool found_min = false;
  for (int i = 0; i < kNumBins; ++i) {
    running += histogram[i];
    cdf[i] = running;
    if (!found_min && running != 0) {
      cdf_min = running;
      found_min = true;
    }
  }

  unsigned char lut[kNumBins];
  const unsigned int total = static_cast<unsigned int>(num_pixels);
  const unsigned int denom = (total > cdf_min) ? (total - cdf_min) : 1u;
  for (int i = 0; i < kNumBins; ++i) {
    const unsigned int numerator =
        (cdf[i] > cdf_min) ? (cdf[i] - cdf_min) : 0u;
    const unsigned int value = (numerator * (kNumBins - 1)) / denom;
    lut[i] = static_cast<unsigned char>(value > 255u ? 255u : value);
  }

  // Step (c): apply the lookup table on the GPU.
  unsigned char* d_lut = nullptr;
  cudaMalloc(&d_lut, kNumBins * sizeof(unsigned char));
  cudaMemcpy(d_lut, lut, kNumBins * sizeof(unsigned char),
             cudaMemcpyHostToDevice);
  ApplyLutKernel<<<grid_size, kBlockSize>>>(d_input, d_output, num_pixels,
                                            d_lut);
  cudaDeviceSynchronize();

  cudaFree(d_lut);
  cudaFree(d_histogram);
}
