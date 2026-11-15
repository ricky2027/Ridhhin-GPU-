// CUDA image-processing driver.
//
// Loads one image or a directory of images, runs the selected GPU kernel(s),
// saves the results, and (optionally) benchmarks GPU against a CPU reference.
//
// Usage:
//   image_filter -i <input> [-o <output_dir>] [-k <kernel>] [-b] [-h]
//     -i <path>    input image, or a directory of images (batch mode)
//     -o <dir>     output directory (default: data/output/)
//     -k <kernel>  grayscale | blur | edge | histeq | negative | all
//     -b           enable CPU-vs-GPU benchmarking
//     -h           print this help

#include <dirent.h>
#include <getopt.h>
#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/kernels.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

// Aborts if a CUDA call returned an error.
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    const cudaError_t status_ = (call);                                       \
    if (status_ != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d '%s': %s\n", __FILE__, __LINE__, \
                   #call, cudaGetErrorString(status_));                       \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

namespace {

constexpr int kNumBins = 256;
constexpr int kLoadChannels = 3;  // Force RGB input.

enum class Kernel { kGrayscale, kBlur, kEdge, kHistEq, kNegative };

struct KernelSpec {
  Kernel id;
  const char* name;
  int out_channels;
};

const KernelSpec kAllKernels[] = {
    {Kernel::kGrayscale, "grayscale", 1},
    {Kernel::kBlur, "blur", 3},
    {Kernel::kEdge, "edge", 1},
    {Kernel::kHistEq, "histeq", 1},
    {Kernel::kNegative, "negative", 3},
};

struct LogRow {
  std::string filename;
  int width;
  int height;
  std::string kernel;
  float gpu_ms;
  float cpu_ms;   // < 0 when benchmarking is disabled.
  float speedup;  // < 0 when benchmarking is disabled.
};

void PrintUsage(const char* prog) {
  std::printf(
      "Usage: %s -i <input> [-o <output_dir>] [-k <kernel>] [-b] [-h]\n"
      "  -i <path>    input image, or a directory of images (batch mode)\n"
      "  -o <dir>     output directory (default: data/output/)\n"
      "  -k <kernel>  grayscale | blur | edge | histeq | negative | all\n"
      "  -b           enable CPU-vs-GPU benchmarking\n"
      "  -h           print this help\n",
      prog);
}

void PrintDeviceInfo() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::fprintf(stderr, "No CUDA-capable device found.\n");
    std::exit(EXIT_FAILURE);
  }
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf("==== Device %d: %s ====\n", device, prop.name);
  std::printf("  Compute capability : %d.%d\n", prop.major, prop.minor);
  std::printf("  Global memory      : %.2f GiB\n",
              static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
  std::printf("  Multiprocessors    : %d\n", prop.multiProcessorCount);
  std::printf("  Max threads/block  : %d\n", prop.maxThreadsPerBlock);
  std::printf("========================================\n\n");
}

// ---------------------------------------------------------------------------
// Filename helpers.
// ---------------------------------------------------------------------------

std::string BaseName(const std::string& path) {
  const size_t slash = path.find_last_of("/\\");
  const std::string name =
      (slash == std::string::npos) ? path : path.substr(slash + 1);
  const size_t dot = name.find_last_of('.');
  return (dot == std::string::npos) ? name : name.substr(0, dot);
}

bool HasImageExtension(const std::string& name) {
  const char* kExts[] = {".png", ".jpg", ".jpeg", ".bmp"};
  std::string lower = name;
  for (char& c : lower) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  for (const char* ext : kExts) {
    const size_t elen = std::strlen(ext);
    if (lower.size() >= elen &&
        lower.compare(lower.size() - elen, elen, ext) == 0) {
      return true;
    }
  }
  return false;
}

bool IsDirectory(const std::string& path) {
  struct stat sb;
  return stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
}

// ---------------------------------------------------------------------------
// CPU reference implementations (used only in benchmark mode).
// ---------------------------------------------------------------------------

inline int ClampInt(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

std::vector<unsigned char> CpuGrayscale(const unsigned char* rgb, int width,
                                        int height) {
  std::vector<unsigned char> out(static_cast<size_t>(width) * height);
  for (int i = 0; i < width * height; ++i) {
    const float r = rgb[i * 3 + 0];
    const float g = rgb[i * 3 + 1];
    const float b = rgb[i * 3 + 2];
    out[i] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b +
                                        0.5f);
  }
  return out;
}

std::vector<unsigned char> CpuNegative(const unsigned char* rgb, int width,
                                       int height) {
  const size_t n = static_cast<size_t>(width) * height * 3;
  std::vector<unsigned char> out(n);
  for (size_t i = 0; i < n; ++i) {
    out[i] = static_cast<unsigned char>(255 - rgb[i]);
  }
  return out;
}

std::vector<unsigned char> CpuGaussianBlur(const unsigned char* rgb, int width,
                                           int height) {
  static const float kWeights[25] = {
      1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6,
      4, 16, 24, 16, 4, 1, 4, 6, 4, 1};
  std::vector<unsigned char> out(static_cast<size_t>(width) * height * 3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        float sum = 0.0f;
        for (int ky = -2; ky <= 2; ++ky) {
          const int sy = ClampInt(y + ky, 0, height - 1);
          for (int kx = -2; kx <= 2; ++kx) {
            const int sx = ClampInt(x + kx, 0, width - 1);
            sum += kWeights[(ky + 2) * 5 + (kx + 2)] *
                   rgb[(sy * width + sx) * 3 + c];
          }
        }
        out[(y * width + x) * 3 + c] =
            static_cast<unsigned char>(sum / 256.0f + 0.5f);
      }
    }
  }
  return out;
}

std::vector<unsigned char> CpuSobel(const unsigned char* rgb, int width,
                                    int height) {
  const std::vector<unsigned char> gray = CpuGrayscale(rgb, width, height);
  static const float kGx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  static const float kGy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  std::vector<unsigned char> out(static_cast<size_t>(width) * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float gx = 0.0f;
      float gy = 0.0f;
      for (int ky = -1; ky <= 1; ++ky) {
        const int sy = ClampInt(y + ky, 0, height - 1);
        for (int kx = -1; kx <= 1; ++kx) {
          const int sx = ClampInt(x + kx, 0, width - 1);
          const int tap = (ky + 1) * 3 + (kx + 1);
          const float v = gray[sy * width + sx];
          gx += kGx[tap] * v;
          gy += kGy[tap] * v;
        }
      }
      const int mag = static_cast<int>(std::sqrt(gx * gx + gy * gy) + 0.5f);
      out[y * width + x] = static_cast<unsigned char>(ClampInt(mag, 0, 255));
    }
  }
  return out;
}

std::vector<unsigned char> CpuHistEq(const unsigned char* rgb, int width,
                                     int height) {
  const std::vector<unsigned char> gray = CpuGrayscale(rgb, width, height);
  const int num_pixels = width * height;
  unsigned int hist[kNumBins] = {0};
  for (int i = 0; i < num_pixels; ++i) {
    ++hist[gray[i]];
  }
  unsigned int cdf[kNumBins];
  unsigned int running = 0;
  unsigned int cdf_min = 0;
  bool found = false;
  for (int i = 0; i < kNumBins; ++i) {
    running += hist[i];
    cdf[i] = running;
    if (!found && running != 0) {
      cdf_min = running;
      found = true;
    }
  }
  unsigned char lut[kNumBins];
  const unsigned int total = static_cast<unsigned int>(num_pixels);
  const unsigned int denom = (total > cdf_min) ? (total - cdf_min) : 1u;
  for (int i = 0; i < kNumBins; ++i) {
    const unsigned int num = (cdf[i] > cdf_min) ? (cdf[i] - cdf_min) : 0u;
    const unsigned int v = (num * (kNumBins - 1)) / denom;
    lut[i] = static_cast<unsigned char>(v > 255u ? 255u : v);
  }
  std::vector<unsigned char> out(static_cast<size_t>(num_pixels));
  for (int i = 0; i < num_pixels; ++i) {
    out[i] = lut[gray[i]];
  }
  return out;
}

std::vector<unsigned char> RunCpu(Kernel kernel, const unsigned char* rgb,
                                  int width, int height, float* cpu_ms) {
  const auto t0 = std::chrono::high_resolution_clock::now();
  std::vector<unsigned char> out;
  switch (kernel) {
    case Kernel::kGrayscale: out = CpuGrayscale(rgb, width, height); break;
    case Kernel::kBlur:      out = CpuGaussianBlur(rgb, width, height); break;
    case Kernel::kEdge:      out = CpuSobel(rgb, width, height); break;
    case Kernel::kHistEq:    out = CpuHistEq(rgb, width, height); break;
    case Kernel::kNegative:  out = CpuNegative(rgb, width, height); break;
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  *cpu_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
  return out;
}

// ---------------------------------------------------------------------------
// GPU execution for a single kernel, timed with CUDA events.
// ---------------------------------------------------------------------------

std::vector<unsigned char> RunGpu(Kernel kernel, const unsigned char* h_rgb,
                                  int width, int height, int out_channels,
                                  float* gpu_ms) {
  const int num_pixels = width * height;
  const size_t color_bytes = static_cast<size_t>(num_pixels) * kLoadChannels;
  const size_t gray_bytes = static_cast<size_t>(num_pixels);
  const size_t out_bytes = static_cast<size_t>(num_pixels) * out_channels;

  unsigned char* d_color = nullptr;
  unsigned char* d_gray = nullptr;
  unsigned char* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_color, color_bytes));
  CUDA_CHECK(cudaMalloc(&d_gray, gray_bytes));
  CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
  CUDA_CHECK(cudaMemcpy(d_color, h_rgb, color_bytes, cudaMemcpyHostToDevice));

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  switch (kernel) {
    case Kernel::kGrayscale:
      LaunchGrayscale(d_color, d_out, width, height, kLoadChannels);
      break;
    case Kernel::kBlur:
      LaunchGaussianBlur(d_color, d_out, width, height, kLoadChannels);
      break;
    case Kernel::kNegative:
      LaunchNegative(d_color, d_out, static_cast<int>(color_bytes));
      break;
    case Kernel::kEdge:
      LaunchGrayscale(d_color, d_gray, width, height, kLoadChannels);
      LaunchSobelEdge(d_gray, d_out, width, height);
      break;
    case Kernel::kHistEq:
      LaunchGrayscale(d_color, d_gray, width, height, kLoadChannels);
      LaunchHistogramEq(d_gray, d_out, width, height);
      break;
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventElapsedTime(gpu_ms, start, stop));

  std::vector<unsigned char> h_out(out_bytes);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_color));
  CUDA_CHECK(cudaFree(d_gray));
  CUDA_CHECK(cudaFree(d_out));
  return h_out;
}

// Processes one image for one kernel: run GPU (and optionally CPU), save the
// output, and append a log row.
void ProcessImage(const std::string& input_path, const std::string& output_dir,
                  const KernelSpec& spec, bool benchmark,
                  std::vector<LogRow>* rows) {
  int width = 0;
  int height = 0;
  int source_channels = 0;
  unsigned char* rgb = stbi_load(input_path.c_str(), &width, &height,
                                 &source_channels, kLoadChannels);
  if (rgb == nullptr) {
    std::fprintf(stderr, "  [skip] cannot load '%s': %s\n", input_path.c_str(),
                 stbi_failure_reason());
    return;
  }

  float gpu_ms = 0.0f;
  std::vector<unsigned char> out =
      RunGpu(spec.id, rgb, width, height, spec.out_channels, &gpu_ms);

  float cpu_ms = -1.0f;
  float speedup = -1.0f;
  if (benchmark) {
    // GPU output is authoritative; the CPU result is computed only to time it.
    std::vector<unsigned char> cpu_out =
        RunCpu(spec.id, rgb, width, height, &cpu_ms);
    (void)cpu_out;
    speedup = (gpu_ms > 0.0f) ? (cpu_ms / gpu_ms) : 0.0f;
  }

  const std::string out_path =
      output_dir + "/" + BaseName(input_path) + "_" + spec.name + ".png";
  if (stbi_write_png(out_path.c_str(), width, height, spec.out_channels,
                     out.data(), width * spec.out_channels) == 0) {
    std::fprintf(stderr, "  [warn] failed to write '%s'\n", out_path.c_str());
  }

  if (benchmark) {
    std::printf("  %-28s %5dx%-5d  %-9s  gpu=%8.3f ms  cpu=%9.3f ms  x%.1f\n",
                BaseName(input_path).c_str(), width, height, spec.name, gpu_ms,
                cpu_ms, speedup);
  } else {
    std::printf("  %-28s %5dx%-5d  %-9s  gpu=%8.3f ms\n",
                BaseName(input_path).c_str(), width, height, spec.name, gpu_ms);
  }

  rows->push_back({BaseName(input_path) + ".png", width, height, spec.name,
                   gpu_ms, cpu_ms, speedup});
}

void WriteCsv(const std::string& output_dir, const std::vector<LogRow>& rows) {
  const std::string path = output_dir + "/execution_log.csv";
  FILE* f = std::fopen(path.c_str(), "w");
  if (f == nullptr) {
    std::fprintf(stderr, "Failed to open '%s' for writing\n", path.c_str());
    return;
  }
  std::fprintf(f, "filename,width,height,kernel,gpu_time_ms,cpu_time_ms,speedup\n");
  for (const LogRow& r : rows) {
    if (r.cpu_ms >= 0.0f) {
      std::fprintf(f, "%s,%d,%d,%s,%.3f,%.3f,%.3f\n", r.filename.c_str(),
                   r.width, r.height, r.kernel.c_str(), r.gpu_ms, r.cpu_ms,
                   r.speedup);
    } else {
      std::fprintf(f, "%s,%d,%d,%s,%.3f,,\n", r.filename.c_str(), r.width,
                   r.height, r.kernel.c_str(), r.gpu_ms);
    }
  }
  std::fclose(f);
  std::printf("\nLog written to %s\n", path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  std::string input_path;
  std::string output_dir = "data/output";
  std::string kernel_arg = "all";
  bool benchmark = false;

  int opt = 0;
  while ((opt = getopt(argc, argv, "i:o:k:bh")) != -1) {
    switch (opt) {
      case 'i': input_path = optarg; break;
      case 'o': output_dir = optarg; break;
      case 'k': kernel_arg = optarg; break;
      case 'b': benchmark = true; break;
      case 'h': PrintUsage(argv[0]); return EXIT_SUCCESS;
      default:  PrintUsage(argv[0]); return EXIT_FAILURE;
    }
  }

  if (input_path.empty()) {
    std::fprintf(stderr, "Error: -i <input> is required.\n\n");
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }

  // Resolve which kernels to run.
  std::vector<KernelSpec> selected;
  if (kernel_arg == "all") {
    for (const KernelSpec& s : kAllKernels) selected.push_back(s);
  } else {
    bool matched = false;
    for (const KernelSpec& s : kAllKernels) {
      if (kernel_arg == s.name) {
        selected.push_back(s);
        matched = true;
        break;
      }
    }
    if (!matched) {
      std::fprintf(stderr, "Error: unknown kernel '%s'.\n\n", kernel_arg.c_str());
      PrintUsage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  // Ensure the output directory exists.
  mkdir(output_dir.c_str(), 0755);

  PrintDeviceInfo();

  // Collect input files (single image or directory batch).
  std::vector<std::string> inputs;
  if (IsDirectory(input_path)) {
    DIR* dir = opendir(input_path.c_str());
    if (dir == nullptr) {
      std::fprintf(stderr, "Error: cannot open directory '%s'.\n",
                   input_path.c_str());
      return EXIT_FAILURE;
    }
    struct dirent* entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
      const std::string name = entry->d_name;
      if (HasImageExtension(name)) {
        inputs.push_back(input_path + "/" + name);
      }
    }
    closedir(dir);
    if (inputs.empty()) {
      std::fprintf(stderr, "No .png/.jpg/.bmp images found in '%s'.\n",
                   input_path.c_str());
      return EXIT_FAILURE;
    }
    std::printf("Batch mode: %zu image(s) in %s\n\n", inputs.size(),
                input_path.c_str());
  } else {
    inputs.push_back(input_path);
  }

  std::vector<LogRow> rows;
  for (const std::string& path : inputs) {
    std::printf("Image: %s\n", path.c_str());
    for (const KernelSpec& spec : selected) {
      ProcessImage(path, output_dir, spec, benchmark, &rows);
    }
  }

  // Summary table.
  std::printf("\n==== Summary (%zu run(s)) ====\n", rows.size());
  if (benchmark) {
    std::printf("%-24s %-9s %12s %12s %8s\n", "file", "kernel", "gpu_ms",
                "cpu_ms", "speedup");
    for (const LogRow& r : rows) {
      std::printf("%-24s %-9s %12.3f %12.3f %7.1fx\n", r.filename.c_str(),
                  r.kernel.c_str(), r.gpu_ms, r.cpu_ms, r.speedup);
    }
  } else {
    std::printf("%-24s %-9s %12s\n", "file", "kernel", "gpu_ms");
    for (const LogRow& r : rows) {
      std::printf("%-24s %-9s %12.3f\n", r.filename.c_str(), r.kernel.c_str(),
                  r.gpu_ms);
    }
  }

  WriteCsv(output_dir, rows);
  return EXIT_SUCCESS;
}
