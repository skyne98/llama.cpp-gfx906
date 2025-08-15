# GFX906 Optimization Project for llama.cpp

## Project Overview

This directory contains comprehensive documentation and implementation guides for optimizing llama.cpp specifically for the AMD Instinct MI50 (gfx906) GPU. The goal is to achieve maximum performance by leveraging hardware-specific features while maintaining a clean, maintainable codebase.

## Documentation Structure

### Core Documents

1. **[optimization_plan.md](optimization_plan.md)**
   - Comprehensive optimization strategy
   - Hardware capability analysis
   - Performance targets and metrics
   - Phased implementation roadmap

2. **[implementation_guide.md](implementation_guide.md)**
   - Detailed kernel implementations
   - Build system modifications
   - Integration with llama.cpp
   - Testing and profiling tools

### Reference Documents

3. **[dev_reference.md](dev_reference.md)**
   - AMD Vega 7nm ISA reference
   - Key instructions for ML/AI workloads
   - Hardware features and capabilities

4. **[matmul.md](matmul.md)**
   - Matrix multiplication strategies
   - Dot product instruction usage
   - Example kernel implementations

5. **[gemini_low_level_review.md](gemini_low_level_review.md)**
   - In-depth GFX906 architecture analysis
   - Memory model and hierarchy
   - AQL packet submission
   - Driver and runtime details

6. **[devin_plan.md](devin_plan.md)**
   - Current llama.cpp support analysis
   - Identified gaps and limitations
   - Integration opportunities

## Quick Start

### Prerequisites

1. **Hardware**: AMD Instinct MI50 (gfx906)
2. **Software**: ROCm 5.7 or compatible version
3. **Build Tools**: CMake 3.14+, HIP compiler

### Building with GFX906 Optimizations

```bash
# Clone the repository
git clone https://github.com/yourusername/llama.cpp-gfx906
cd llama.cpp-gfx906

# Build with GFX906 optimizations
cmake -B build \
  -DGGML_HIP=ON \
  -DGGML_HIP_GFX906_OPTIMIZED=ON \
  -DAMDGPU_TARGETS=gfx906 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)
```

### Running Benchmarks

```bash
# Basic inference benchmark
./build/bin/llama-bench \
  -m models/llama-7b-q4_0.gguf \
  -p 512 \
  -n 128 \
  -t 1

# Profile with rocprof
rocprof --stats --hip-trace \
  ./build/bin/llama-cli \
  -m models/llama-7b-q4_0.gguf \
  -p "Once upon a time" \
  -n 100
```

## Key Optimizations

### 1. Hardware-Specific Instructions

- **V_DOT4_I32_I8**: 4x INT8 dot products for quantized models
- **V_DOT2_F32_F16**: 2x FP16 dot products for mixed precision
- **V_PK_FMA_F16**: Dual FP16 FMA operations
- **DS_PERMUTE/BPERMUTE**: Hardware lane shuffling

### 2. Memory Hierarchy Optimization

- **64KB LDS**: Full utilization of Local Data Share
- **Coalesced Access**: 128-byte aligned memory patterns
- **Double Buffering**: Overlap compute with memory transfers
- **HBM2 Bandwidth**: ~1TB/s effective utilization

### 3. Wave-Level Programming

- **64-thread waves**: GCN-specific optimizations
- **Wave reductions**: Efficient butterfly patterns
- **Lane shuffles**: Hardware-accelerated data exchange

### 4. Kernel Specialization

- **Quantization-aware**: Optimized for Q4_0, Q8_0, Q5_K
- **Tile sizes**: Tuned for 60 Compute Units
- **Occupancy**: Maximized wave utilization

## Performance Expectations

| Component | Expected Improvement |
|-----------|--------------------|
| Matrix Multiplication | 30-40% |
| Attention Mechanism | 25-35% |
| Quantized Operations | 40-50% |
| Memory Bandwidth | 85-90% utilization |
| **Overall Inference** | **35-45%** |

## Testing

### Unit Tests
```bash
# Run GFX906-specific tests
ctest -L gfx906
```

### Validation
```bash
# Compare with reference implementation
./scripts/validate_gfx906.sh
```

### Performance Analysis
```bash
# Detailed performance metrics
./scripts/profile_gfx906.sh
```

## Development Workflow

1. **Feature Branch**: Create feature branch for optimizations
2. **Implementation**: Follow implementation_guide.md
3. **Testing**: Run unit tests and validation
4. **Profiling**: Analyze performance with rocprof
5. **Optimization**: Iterate based on metrics
6. **Integration**: Merge into main branch

## Troubleshooting

### Common Issues

1. **Compilation Errors**
   - Ensure ROCm 5.7 is installed
   - Check AMDGPU_TARGETS is set to gfx906
   - Verify HIP compiler version

2. **Runtime Errors**
   - Check GPU is properly detected: `rocminfo`
   - Verify kernel modules: `lsmod | grep amdgpu`
   - Monitor GPU: `rocm-smi`

3. **Performance Issues**
   - Profile with rocprof
   - Check occupancy metrics
   - Verify memory access patterns

## Contributing

Contributions are welcome! Please:

1. Follow the coding standards in implementation_guide.md
2. Add tests for new kernels
3. Profile and document performance improvements
4. Update documentation as needed

## Resources

- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [LLVM AMDGPU Backend](https://llvm.org/docs/AMDGPUUsage.html)
- [HSA Runtime](http://www.hsafoundation.com/)
- [AMD ISA Documentation](https://gpuopen.com/amd-isa-documentation/)

## License

This project maintains the same license as the original llama.cpp project.

## Acknowledgments

- Original llama.cpp contributors
- AMD ROCm team
- Community members who provided hardware access and testing

---

*Last Updated: 2024*
*Target Hardware: AMD Instinct MI50 (gfx906)*
*ROCm Version: 5.7*