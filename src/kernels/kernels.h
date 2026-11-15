#ifndef SRC_KERNELS_KERNELS_H_
#define SRC_KERNELS_KERNELS_H_

// Host-side launcher wrappers for the CUDA image-processing kernels.
//
// All pointers passed to these functions are DEVICE pointers unless noted
// otherwise. Each launcher configures the grid/block dimensions, launches its
// __global__ kernel, and returns once the launch has been issued. Callers are
// responsible for device allocation, host<->device transfers, and
// synchronization (e.g. cudaDeviceSynchronize) unless a launcher documents
// that it synchronizes internally.

#ifdef __cplusplus
extern "C" {
#endif

// Converts an interleaved RGB(A) image to single-channel grayscale using the
// ITU-R BT.601 luma weights. `d_input` holds `width * height * channels`
// bytes; `d_output` holds `width * height` bytes.
void LaunchGrayscale(const unsigned char* d_input, unsigned char* d_output,
                     int width, int height, int channels);

// Applies a 5x5 Gaussian blur (1/256 weighted kernel) to an interleaved image
// with `channels` channels. Borders are clamped. `d_input` and `d_output` each
// hold `width * height * channels` bytes and must not alias.
void LaunchGaussianBlur(const unsigned char* d_input, unsigned char* d_output,
                        int width, int height, int channels);

// Runs Sobel edge detection on a single-channel grayscale image. Output is the
// gradient magnitude clamped to [0, 255]. Both buffers hold `width * height`
// bytes.
void LaunchSobelEdge(const unsigned char* d_input, unsigned char* d_output,
                     int width, int height);

// Performs histogram equalization on a single-channel grayscale image. The
// histogram and remap are computed on the GPU; the CDF is computed on the host.
// Both buffers hold `width * height` bytes. This launcher synchronizes
// internally because of the intermediate device<->host transfers.
void LaunchHistogramEq(const unsigned char* d_input, unsigned char* d_output,
                       int width, int height);

// Produces the photographic negative (255 - value) of every byte in the image.
// Operates on all channels; `num_bytes` is the total byte count
// (`width * height * channels`).
void LaunchNegative(const unsigned char* d_input, unsigned char* d_output,
                    int num_bytes);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SRC_KERNELS_KERNELS_H_
