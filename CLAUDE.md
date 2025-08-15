# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
llama.cpp-gfx906 is a high-performance C/C++ implementation for LLM inference with AMD GFX906 GPU support. This is a specialized fork focusing on AMD GPU architecture.

## Build Commands

### Standard CPU Build
```bash
cmake -B build
cmake --build build --config Release
```

### AMD GPU Build (GFX906)
```bash
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906
cmake --build build --config Release
```

## Testing

### Run All Tests
```bash
cmake -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build --config Release
cd build && ctest
```

### Run Specific Test Categories
```bash
ctest -L main     # Main functionality
ctest -L model    # Model loading
```

### Run Individual Tests
```bash
./build/bin/test-backend-ops
./build/bin/test-quantize-fns
./build/bin/test-tokenizer-0 ./models/ggml-vocab-llama-bpe.gguf
```

## Code Formatting
Use clang-format for all C/C++ code. The repository follows 4-space indentation (configured in .ecrc).

## Architecture

### Layer Structure
1. **GGML Layer** (`ggml/`): Low-level tensor operations and backend implementations
   - `ggml/src/ggml.c`: Core tensor library
   - `ggml/src/ggml-cuda/`: NVIDIA GPU kernels
   - `ggml/src/ggml-hip/`: AMD GPU kernels
   - `ggml/src/ggml-backend.c`: Backend abstraction layer

2. **LLaMA Layer** (`src/`): Model implementation and inference engine
   - `src/llama.cpp`: Main inference engine - coordinates model loading, context management, and inference
   - `src/llama-model.*`: Model format handling and weight loading
   - `src/llama-vocab.*`: Tokenization across different vocab types (BPE, SPM, etc.)
   - `src/llama-sampling.*`: Sampling strategies (greedy, top-k, top-p, etc.)

3. **Tools Layer** (`tools/`): User-facing applications
   - `tools/main/`: CLI tool for model inference
   - `tools/server/`: HTTP server with OpenAI API compatibility
   - `tools/quantize/`: Model quantization utilities

### Key Design Patterns
- **Backend Abstraction**: All compute operations go through ggml-backend interface, allowing seamless switching between CPU/CUDA/HIP/Vulkan
- **Model Format**: Uses GGUF (GGML Universal Format) for model storage with metadata and tensor data
- **Memory Management**: Custom allocators with mmap support for efficient large model loading
- **Quantization**: Supports multiple quantization levels (Q4_0, Q5_K_M, etc.) defined in `ggml/include/ggml.h`

## Development Guidelines

### Adding New Features
- Model architecture additions go in `src/llama.cpp` (search for `llm_load_arch`)
- New sampling methods belong in `src/llama-sampling.cpp`
- Backend kernels should be added to respective backend directories under `ggml/src/`

### Before Committing
1. Run clang-format on modified files
2. Build with tests enabled and run ctest
3. Test with both CPU and GPU builds if modifying backend code
4. Check performance impact with perplexity tool

### Common Development Tasks
- **Add new model architecture**: Modify `llm_load_arch()` and `llm_build_*()` functions in `src/llama.cpp`
- **Implement new operator**: Add to `ggml/src/ggml.c` and implement in relevant backends
- **Add sampling method**: Extend `src/llama-sampling.cpp` with new sampling strategy
- **Debug tokenization**: Use `tools/test-tokenizer-*.cpp` utilities

## Important Configuration
- C++17 required
- CMake 3.14+ required
- For AMD GPU: ROCm toolkit and HIP compiler required
- Environment variables:
  - `HIP_VISIBLE_DEVICES`: Control AMD GPU visibility
  - `CUDA_VISIBLE_DEVICES`: Control NVIDIA GPU visibility
  - `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`: Enable unified memory for CUDA