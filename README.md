# GPU-Accelerated Batch Image Processing Pipeline

## Description

This project is a command-line image processing pipeline that runs a set of
classic image filters on the GPU using **CUDA**, with batch processing over a
whole directory of images and an optional **CPU-vs-GPU benchmark** mode.

The goal is twofold:

1. **Throughput** — process many images automatically (batch mode), applying
   one or all kernels to each.
2. **Comparison** — measure how much faster the data-parallel GPU
   implementation is than an equivalent single-threaded CPU implementation,
   and record the results to a CSV for analysis.

Image I/O uses the header-only [stb](https://github.com/nothings/stb)
libraries, so there are no external image-library dependencies — just the CUDA
toolkit and a C++17 compiler.

## Code Organization

```
.
├── bin/                     Built executable (image_processor) — git-ignored
├── data/
│   ├── input/               Source images (fed to the pipeline)
│   └── output/              Generated images + execution_log.csv — git-ignored
├── lib/
│   ├── stb_image.h          Header-only PNG/JPG/BMP decoder (public domain)
│   └── stb_image_write.h    Header-only PNG/JPG encoder (public domain)
├── src/
│   ├── main.cu              Driver: CLI parsing, batch loop, timing, logging
│   └── kernels/
│       ├── kernels.h        Declarations of all Launch* host wrappers
│       ├── grayscale.cu     RGB -> grayscale kernel + launcher
│       ├── gaussian_blur.cu 5x5 Gaussian blur kernel + launcher
│       ├── sobel_edge.cu    Sobel edge-detection kernel + launcher
│       ├── histogram_eq.cu  Histogram-equalization kernels + launcher
│       └── negative.cu      Image inversion kernel + launcher
├── Makefile                 nvcc build (targets: all, run, clean)
├── run.sh                   One-command demo: build, fetch samples, run
├── LICENSE                  MIT
└── README.md                This file
```

### File details

- **`src/main.cu`** — parses CLI flags, discovers input files (single image or
  a directory via `dirent.h`), loads each with stb, allocates device memory,
  dispatches the selected kernel(s), times them with CUDA events, copies the
  result back, writes it out, and logs to `execution_log.csv`. In benchmark
  mode it also runs CPU reference implementations and reports the speedup. A
  `CUDA_CHECK()` macro guards every CUDA API call, and device properties are
  printed at startup.
- **`src/kernels/*.cu`** — one translation unit per filter. Each contains a
  `__global__` kernel and a `Launch*` host wrapper that configures the grid and
  block dimensions. Convolution weights (Gaussian, Sobel) live in
  `__constant__` memory.
- **`src/kernels/kernels.h`** — the shared interface `main.cu` links against.

## Supported Kernels

| Kernel      | CLI name    | Algorithm |
|-------------|-------------|-----------|
| Grayscale   | `grayscale` | Per-pixel luma using ITU-R BT.601 weights: `0.299R + 0.587G + 0.114B`. 1D grid, block size 256. |
| Gaussian Blur | `blur`    | 5×5 convolution with the standard 1/256 Gaussian kernel (weights in constant memory), edge-clamped. 2D grid, 16×16 blocks. |
| Sobel Edge  | `edge`      | Converts to grayscale, then applies 3×3 Sobel Gx/Gy (constant memory) and outputs gradient magnitude `sqrt(gx² + gy²)` clamped to 255. 2D grid. |
| Histogram Eq | `histeq`   | Grayscale, then GPU histogram via `atomicAdd` → CPU CDF over 256 bins → GPU lookup-table remap for contrast stretching. |
| Negative    | `negative`  | Per-byte inversion `255 - value` across all channels. 1D grid, block size 256. |

Use `-k all` to run every kernel on each image.

## Prerequisites

- **CUDA Toolkit 11+** (`nvcc` on your `PATH`)
- A **CUDA-capable NVIDIA GPU**
- **Linux or WSL** (CUDA is not supported on macOS)
- **make**
- *Optional, for `run.sh` sample fetching:* `curl` or `wget`, and ImageMagick
  (`magick`/`convert`) to convert the SIPI TIFF samples to PNG.

## Building

```bash
make          # or: make all
```

This compiles every `.cu` under `src/` and `src/kernels/` with
`nvcc -O2 -std=c++17` and links them into `bin/image_processor`.

Target a specific GPU architecture by overriding `GPU_ARCH`:

```bash
make GPU_ARCH=sm_75
```

Remove the build with:

```bash
make clean
```

## Usage

```
./bin/image_processor -i <input> [-o <output_dir>] [-k <kernel>] [-b] [-h]
```

| Flag | Description |
|------|-------------|
| `-i <path>` | Input image **or** a directory of images (enables batch mode). **Required.** |
| `-o <dir>`  | Output directory. Default: `data/output/`. |
| `-k <kernel>` | `grayscale`, `blur`, `edge`, `histeq`, `negative`, or `all`. Default: `all`. |
| `-b` | Benchmark mode: also run the CPU reference and report the speedup. |
| `-h` | Print usage help. |

Outputs are named `<original>_<kernel>.png` in the output directory.

Examples:

```bash
# Single image, single kernel
./bin/image_processor -i data/input/baboon.png -k grayscale

# Single image, every kernel
./bin/image_processor -i data/input/baboon.png -k all

# Whole directory, every kernel, with CPU-vs-GPU timing
./bin/image_processor -i data/input/ -o data/output/ -k all -b
```

## Running

**Quick demo** (builds if needed, fetches sample images, runs everything):

```bash
bash run.sh
```

**Manual:**

```bash
make
./bin/image_processor -i data/input/ -o data/output/ -k all -b
```

On start the program prints the GPU name, compute capability, and memory. For
each image it prints the filename, dimensions, kernel, and GPU time (plus CPU
time and speedup in `-b` mode), then a summary table. It also writes
`data/output/execution_log.csv` with columns:

```
filename, width, height, kernel, gpu_time_ms, cpu_time_ms, speedup
```

## Sample Results

> _Placeholder — fill in after running on a real GPU._

Example of the kind of table to paste here (from `execution_log.csv` or the
on-screen summary):

| File        | Kernel    | GPU (ms) | CPU (ms) | Speedup |
|-------------|-----------|---------:|---------:|--------:|
| baboon.png  | grayscale |     _tbd_ |    _tbd_ |   _tbd_ |
| baboon.png  | blur      |     _tbd_ |    _tbd_ |   _tbd_ |
| baboon.png  | edge      |     _tbd_ |    _tbd_ |   _tbd_ |
| baboon.png  | histeq    |     _tbd_ |    _tbd_ |   _tbd_ |
| baboon.png  | negative  |     _tbd_ |    _tbd_ |   _tbd_ |

Before/after images: the originals live in `data/input/` and the processed
results in `data/output/` as `<name>_<kernel>.png`. _(Embed a couple of
before/after pairs here.)_

## Performance

In benchmark mode (`-b`) each filter is run twice per image: once on the GPU
(timed with `cudaEvent_t` start/stop around the kernel launches) and once with
a single-threaded CPU reference (timed with `std::chrono`). The speedup column
is `cpu_time_ms / gpu_time_ms`.

What to expect:

- **Compute-heavy, highly parallel kernels** (Gaussian blur, Sobel) usually
  show the largest speedups, because every output pixel is independent and the
  GPU runs thousands of them concurrently.
- **Memory-bound, lightweight kernels** (negative, grayscale) are limited by
  bandwidth; they still speed up, but by less, since there is little arithmetic
  to hide the transfer cost.
- **Histogram equalization** includes a serial CDF step on the CPU and extra
  device↔host transfers, so its end-to-end speedup is typically more modest.

Note that the reported GPU time measures the **kernel launches only**, not the
host↔device memory copies. For small images the copy overhead can dominate the
total wall-clock time, so GPU wins are most dramatic on larger images where
compute outweighs transfer.

## Lessons Learned

> _Placeholder — fill in with your own observations, e.g.:_
>
> - Where constant memory / block sizes mattered for performance.
> - Surprises in the CPU-vs-GPU gap across kernels and image sizes.
> - The impact of H2D/D2H transfers on real-world speedup.
> - Anything you'd do differently (shared memory tiling, streams, pinned
>   memory, etc.).

## License

MIT — see [LICENSE](LICENSE).
