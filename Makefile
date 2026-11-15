# Build for the GPU-accelerated batch image processing pipeline.
#
#   make / make all   # build bin/image_processor
#   make run          # build and run with sample args (-i data/input -k all -b)
#   make clean        # remove build artifacts

NVCC      ?= nvcc
# Override on the command line, e.g. `make GPU_ARCH=sm_75`.
GPU_ARCH  ?= sm_70
NVCCFLAGS ?= -O2 -std=c++17 -arch=$(GPU_ARCH)

# stb headers live in lib/; kernel headers in src/kernels/; main includes
# "kernels/kernels.h" relative to src/ and stb relative to ../lib/.
INCLUDES  := -Isrc -Isrc/kernels -Ilib

SRC_DIR    := src
KERNEL_DIR := $(SRC_DIR)/kernels
BIN_DIR    := bin
TARGET     := $(BIN_DIR)/image_processor

# All CUDA translation units from src/ and src/kernels/.
SOURCES := $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(KERNEL_DIR)/*.cu)

# Sample run configuration (override on the command line).
RUN_KERNEL ?= all
RUN_INPUT  ?= data/input
RUN_OUTPUT ?= data/output
RUN_FLAGS  ?= -b

.PHONY: all run clean

all: $(TARGET)

# Single nvcc invocation compiles every .cu and links them together, so the
# __constant__ symbols and Launch* wrappers resolve without separate steps.
$(TARGET): $(SOURCES) $(KERNEL_DIR)/kernels.h | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SOURCES) -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

run: $(TARGET)
	./$(TARGET) -i $(RUN_INPUT)/ -o $(RUN_OUTPUT)/ -k $(RUN_KERNEL) $(RUN_FLAGS)

clean:
	rm -f $(TARGET)
