#!/bin/bash
# Create GitHub issues for GFX906 optimization project
# Requires: gh CLI tool authenticated with your repository

set -e

# Configuration
REPO="skyne98/llama.cpp-gfx906"  # Update with your repo
PROJECT="GFX906 Optimization"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}📋 Creating GitHub Issues for GFX906 Optimization Project${NC}"
echo -e "${YELLOW}Repository: $REPO${NC}"
echo ""

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed."
    echo "Install it from: https://cli.github.com/"
    exit 1
fi

# Check authentication
if ! gh auth status &> /dev/null; then
    echo "Error: Not authenticated with GitHub."
    echo "Run: gh auth login"
    exit 1
fi

# Create labels if they don't exist
echo -e "${GREEN}Creating labels...${NC}"
gh label create "gfx906" --description "AMD Instinct MI50 specific" --color "FF6B6B" 2>/dev/null || true
gh label create "optimization" --description "Performance optimization" --color "4ECDC4" 2>/dev/null || true
gh label create "kernel" --description "GPU kernel implementation" --color "45B7D1" 2>/dev/null || true
gh label create "build" --description "Build system and configuration" --color "96CEB4" 2>/dev/null || true
gh label create "testing" --description "Testing and validation" --color "FFEAA7" 2>/dev/null || true
gh label create "memory" --description "Memory optimization" --color "DDA0DD" 2>/dev/null || true
gh label create "foundation" --description "Foundation work" --color "98D8C8" 2>/dev/null || true

# Create milestones
echo -e "${GREEN}Creating milestones...${NC}"
gh api repos/$REPO/milestones -f title="Phase 1: Foundation" -f description="Build system, Docker setup, and basic infrastructure" -f due_on="2024-02-15T00:00:00Z" 2>/dev/null || true
gh api repos/$REPO/milestones -f title="Phase 2: Core Kernels" -f description="Implement optimized kernels for matrix multiplication and attention" -f due_on="2024-03-01T00:00:00Z" 2>/dev/null || true
gh api repos/$REPO/milestones -f title="Phase 3: Memory Optimization" -f description="Optimize memory access patterns and LDS usage" -f due_on="2024-03-15T00:00:00Z" 2>/dev/null || true
gh api repos/$REPO/milestones -f title="Phase 4: Testing & Validation" -f description="Comprehensive testing and performance validation" -f due_on="2024-03-30T00:00:00Z" 2>/dev/null || true

echo ""
echo -e "${BLUE}Creating issues...${NC}"
echo ""

# ============================================================================
# PHASE 1: FOUNDATION ISSUES
# ============================================================================

echo -e "${GREEN}Phase 1: Foundation Issues${NC}"

# Issue 1: Docker Environment Setup
gh issue create \
  --title "Set up Docker development environment for GFX906" \
  --body "## Description
Create a Docker-based development environment optimized for AMD Instinct MI50 (gfx906) GPU development.

## Acceptance Criteria
- [ ] Dockerfile with ROCm 5.7.3 base image
- [ ] docker-compose.yml with proper GPU passthrough
- [ ] Development and runtime stages
- [ ] ccache integration for fast rebuilds
- [ ] Verification script to check GPU access
- [ ] Documentation in docs/gfx906/docker_setup.md

## Technical Details
- Use \`rocm/dev-ubuntu-22.04:5.7.3-complete\` as base
- Set \`HSA_OVERRIDE_GFX_VERSION=9.0.6\`
- Configure GPU devices: \`/dev/kfd\`, \`/dev/dri\`
- Add video and render groups
- Set IPC mode to host for multi-process GPU apps

## References
- [Docker setup documentation](docs/gfx906/docker_setup.md)
- [ROCm Docker documentation](https://rocm.docs.amd.com/en/latest/deploy/docker.html)

## Testing
\`\`\`bash
# Verify GPU access in container
docker compose run gfx906-dev rocminfo | grep gfx906
\`\`\`" \
  --label "foundation,build,gfx906" \
  --milestone "Phase 1: Foundation"

# Issue 2: Build System Configuration
gh issue create \
  --title "Configure CMake build system for GFX906 optimizations" \
  --body "## Description
Set up CMake configuration with GFX906-specific compilation flags and optimization settings.

## Acceptance Criteria
- [ ] CMakeLists.txt modifications for GGML_HIP_GFX906_OPTIMIZED flag
- [ ] Conditional compilation paths for gfx906
- [ ] Architecture-specific compiler flags
- [ ] Separate build targets for optimized kernels
- [ ] Integration with existing GGML build system

## Implementation Details
\`\`\`cmake
if(GGML_HIP AND GGML_HIP_GFX906_OPTIMIZED)
    set(AMDGPU_TARGETS \"gfx906\" CACHE STRING \"AMD GPU targets\")
    add_compile_definitions(GGML_HIP_GFX906_OPTIMIZED)
    list(APPEND HIP_CXX_FLAGS 
        -mwavefrontsize64
        -mcumode
        -ffast-math)
endif()
\`\`\`

## References
- [Implementation guide](docs/gfx906/implementation_guide.md#build-system-modifications)
- LLVM AMDGPU backend documentation

## Testing
- Build with \`-DGGML_HIP_GFX906_OPTIMIZED=ON\`
- Verify gfx906-specific code paths are compiled
- Check symbol presence with \`nm\`" \
  --label "foundation,build,gfx906" \
  --milestone "Phase 1: Foundation"

# Issue 3: Hardware Detection and Dispatch
gh issue create \
  --title "Implement runtime hardware detection and kernel dispatch system" \
  --body "## Description
Create a runtime detection system to identify GFX906 hardware and dispatch to optimized kernels.

## Acceptance Criteria
- [ ] Runtime GPU architecture detection
- [ ] Kernel dispatch mechanism
- [ ] Fallback to generic kernels when not on gfx906
- [ ] Performance impact < 0.1% from dispatch overhead
- [ ] Unit tests for detection logic

## Implementation
\`\`\`cpp
static inline bool is_gfx906() {
    hipDeviceProp_t prop;
    CUDA_CHECK(hipGetDeviceProperties(&prop, 0));
    return prop.gcnArch == 906;
}

template<typename KernelFunc, typename FallbackFunc>
__host__ void dispatch_gfx906(KernelFunc gfx906_kernel, 
                              FallbackFunc fallback_kernel,
                              dim3 grid, dim3 block, ...) {
    if (is_gfx906()) {
        gfx906_kernel<<<grid, block>>>(...);  
    } else {
        fallback_kernel<<<grid, block>>>(...);  
    }
}
\`\`\`

## References
- [Implementation guide](docs/gfx906/implementation_guide.md#kernel-dispatch-system)
- HIP runtime API documentation" \
  --label "foundation,kernel,gfx906" \
  --milestone "Phase 1: Foundation"

# ============================================================================
# PHASE 2: KERNEL OPTIMIZATION ISSUES
# ============================================================================

echo -e "${GREEN}Phase 2: Kernel Optimization Issues${NC}"

# Issue 4: DP4A Instruction Implementation
gh issue create \
  --title "Implement optimized DP4A (dot product) instructions for INT8 operations" \
  --body "## Description
Implement hardware-accelerated dot product instructions (V_DOT4_I32_I8) for quantized model inference.

## Acceptance Criteria
- [ ] Native V_DOT4_I32_I8 instruction wrapper
- [ ] Native V_DOT2_F32_F16 instruction wrapper
- [ ] Native V_DOT8_I32_U4 for INT4 quantization
- [ ] Performance test showing >2x speedup vs scalar
- [ ] Correctness validation against reference

## Implementation
\`\`\`cpp
// V_DOT4_I32_I8 - 4x INT8 dot product
__device__ __forceinline__ int32_t dot4_i8_gfx906(
    const int32_t a,  // packed 4x int8
    const int32_t b,  // packed 4x int8
    const int32_t c   // accumulator
) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// V_DOT2_F32_F16 - 2x FP16 dot product  
__device__ __forceinline__ float dot2_f16_gfx906(
    const uint32_t a,  // packed 2x fp16
    const uint32_t b,  // packed 2x fp16
    const float c      // accumulator
) {
    return __builtin_amdgcn_fdot2(a, b, c, false);
}
\`\`\`

## Performance Targets
- INT8 GEMM: >100 TFLOPS
- FP16 GEMM: >50 TFLOPS
- Memory bandwidth: >900 GB/s

## References
- [AMD Vega ISA Reference](docs/gfx906/dev_reference.md)
- [Matrix multiplication strategies](docs/gfx906/matmul.md)
- LLVM builtin documentation

## Testing
\`\`\`cpp
TEST(GFX906, DotProduct) {
    // Test accuracy
    // Test performance
    // Test edge cases
}
\`\`\`" \
  --label "kernel,optimization,gfx906" \
  --milestone "Phase 2: Core Kernels"

# Issue 5: Optimized Matrix Multiplication Kernel
gh issue create \
  --title "Implement optimized GEMM kernel for Q8_0 quantization" \
  --body "## Description
Create a highly optimized matrix multiplication kernel specifically tuned for GFX906's 60 compute units.

## Acceptance Criteria
- [ ] Tile sizes optimized for 64KB LDS
- [ ] Efficient use of V_DOT4_I32_I8 instructions
- [ ] Double buffering for memory transfers
- [ ] >35% speedup vs generic implementation
- [ ] Support for all quantization types (Q4_0, Q8_0, Q5_K)

## Key Optimizations
- Tile size: 128x128x32 (tuned for 60 CUs)
- 4 waves per block (256 threads)
- Full LDS utilization (64KB)
- Coalesced memory access patterns
- Async memory copies overlapped with compute

## Implementation Structure
\`\`\`cpp
template<int TILE_M=128, int TILE_N=128, int TILE_K=32>
__global__ void gemm_q8_0_gfx906(
    const block_q8_0* __restrict__ A,
    const block_q8_0* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    __shared__ int8_t tile_a[TILE_M][TILE_K + 4];  // +4 for bank conflicts
    __shared__ int8_t tile_b[TILE_K][TILE_N + 4];
    // Implementation...
}
\`\`\`

## Performance Metrics
- Target: 85-90% of theoretical peak
- Measure: tokens/second improvement
- Profile: occupancy, memory efficiency

## References
- [Implementation guide](docs/gfx906/implementation_guide.md#optimized-matrix-multiplication)
- [GFX906 architecture details](docs/gfx906/gemini_low_level_review.md)" \
  --label "kernel,optimization,gfx906" \
  --milestone "Phase 2: Core Kernels"

# Issue 6: Flash Attention Implementation
gh issue create \
  --title "Implement Flash Attention optimized for GFX906 architecture" \
  --body "## Description
Implement memory-efficient attention mechanism optimized for GFX906's memory hierarchy.

## Acceptance Criteria
- [ ] Tiled attention computation fitting in LDS
- [ ] Online softmax implementation
- [ ] Support for causal masking
- [ ] Memory usage O(N) instead of O(N²)
- [ ] 25-35% speedup vs baseline

## Technical Details
- Block size tuned for 64KB LDS
- Use V_PK_FMA_F16 for dual FP16 operations
- DS_PERMUTE for efficient transposes
- Wave-level reductions for softmax

## Implementation Approach
\`\`\`cpp
template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__global__ void flash_attn_f16_gfx906(
    const half* Q, const half* K, const half* V,
    half* O, const float scale,
    const int batch, const int seqlen, const int nheads
) {
    // Shared memory for Q, K, V tiles
    extern __shared__ char smem[];
    // Tiled computation with online softmax
}
\`\`\`

## References
- [Flash Attention paper](https://arxiv.org/abs/2205.14135)
- [Implementation guide](docs/gfx906/implementation_guide.md#optimized-attention-kernel)" \
  --label "kernel,optimization,gfx906" \
  --milestone "Phase 2: Core Kernels"

# ============================================================================
# PHASE 3: MEMORY OPTIMIZATION ISSUES
# ============================================================================

echo -e "${GREEN}Phase 3: Memory Optimization Issues${NC}"

# Issue 7: LDS Memory Optimization
gh issue create \
  --title "Optimize Local Data Share (LDS) usage for maximum throughput" \
  --body "## Description
Maximize utilization of the 64KB LDS memory per compute unit for improved data reuse.

## Acceptance Criteria
- [ ] Full 64KB LDS utilization in key kernels
- [ ] Bank conflict avoidance strategies
- [ ] Double buffering implementation
- [ ] Measured >80% LDS efficiency
- [ ] Documentation of LDS layout patterns

## Optimization Strategies
1. **Padding for bank conflicts**: Add padding to avoid 32-bank conflicts
2. **Data layout**: Optimize for coalesced access patterns
3. **Double buffering**: Overlap computation with data movement
4. **Swizzling**: Use address swizzling for conflict-free access

## Implementation
\`\`\`cpp
// Optimized LDS allocation
template<typename T, int ROWS, int COLS>
struct LDSTile {
    static constexpr int BANK_WIDTH = 32;
    static constexpr int PAD = 4;  // Avoid bank conflicts
    __shared__ T data[ROWS][COLS + PAD];
    
    __device__ void load_from_global(const T* gmem, int stride) {
        // Coalesced load implementation
    }
};
\`\`\`

## References
- [Memory optimization plan](docs/gfx906/optimization_plan.md#memory-hierarchy-optimization)
- AMD LDS optimization guide" \
  --label "memory,optimization,gfx906" \
  --milestone "Phase 3: Memory Optimization"

# Issue 8: Coalesced Memory Access Patterns
gh issue create \
  --title "Implement coalesced global memory access patterns" \
  --body "## Description
Optimize global memory access patterns for maximum bandwidth utilization on HBM2.

## Acceptance Criteria
- [ ] 128-byte aligned memory accesses
- [ ] Vector load/store instructions (dwordx4)
- [ ] Memory access coalescing analysis
- [ ] >85% memory bandwidth utilization
- [ ] Profiling results showing improvement

## Implementation Techniques
\`\`\`cpp
namespace gfx906 {
// Vectorized load with alignment
template<typename T>
__device__ __forceinline__ void load_vectorized(
    T* dst, const T* __restrict__ src, int count
) {
    // Check 128-byte alignment
    if (((uintptr_t)src & 15) == 0) {
        // Use float4 loads for 128-bit access
        #pragma unroll 4
        for (int i = threadIdx.x; i < count/4; i += blockDim.x) {
            float4 data = ((const float4*)src)[i];
            ((float4*)dst)[i] = data;
        }
    }
}
}
\`\`\`

## Performance Targets
- Read bandwidth: >900 GB/s (90% of theoretical)
- Write bandwidth: >850 GB/s
- L2 cache hit rate: >60%

## References
- [Implementation guide](docs/gfx906/implementation_guide.md#memory-access-optimization)
- HBM2 specifications" \
  --label "memory,optimization,gfx906" \
  --milestone "Phase 3: Memory Optimization"

# Issue 9: Wave-Level Primitives
gh issue create \
  --title "Implement efficient wave-level reduction and shuffle operations" \
  --body "## Description
Create optimized wave-level primitives using GCN's 64-thread wave architecture.

## Acceptance Criteria
- [ ] Wave reduction (sum, max, min)
- [ ] Wave broadcast operations
- [ ] Wave shuffle/permute operations
- [ ] Prefix sum implementation
- [ ] Performance comparison with shared memory approach

## Implementation
\`\`\`cpp
namespace gfx906 {
// Butterfly reduction across 64-thread wave
template<typename T, typename Op>
__device__ __forceinline__ T wave_reduce(T value, Op op) {
    #pragma unroll
    for (int offset = 32; offset >= 1; offset >>= 1) {
        T other = __builtin_amdgcn_ds_swizzle(
            value, 0x1F, offset  // XOR swizzle
        );
        value = op(value, other);
    }
    return value;
}

// Broadcast from lane 0
template<typename T>
__device__ __forceinline__ T wave_broadcast(T value) {
    return __builtin_amdgcn_readfirstlane(value);
}
}
\`\`\`

## Performance Benefits
- 10x faster than shared memory reductions
- No LDS usage required
- Single-cycle latency

## References
- [AMD GCN ISA documentation](docs/gfx906/dev_reference.md)
- [Implementation guide](docs/gfx906/implementation_guide.md#wave-level-primitives)" \
  --label "kernel,optimization,gfx906" \
  --milestone "Phase 3: Memory Optimization"

# ============================================================================
# PHASE 4: TESTING AND VALIDATION ISSUES
# ============================================================================

echo -e "${GREEN}Phase 4: Testing and Validation Issues${NC}"

# Issue 10: Unit Test Framework
gh issue create \
  --title "Create comprehensive unit test framework for GFX906 kernels" \
  --body "## Description
Develop a testing framework to validate correctness and performance of GFX906-specific optimizations.

## Acceptance Criteria
- [ ] Unit tests for all custom kernels
- [ ] Accuracy validation against reference implementation
- [ ] Performance regression tests
- [ ] Edge case and boundary testing
- [ ] Automated test execution in CI/CD

## Test Structure
\`\`\`cpp
class GFX906KernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check for gfx906 hardware
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, 0);
        if (prop.gcnArch != 906) {
            GTEST_SKIP() << \"Not running on gfx906\";
        }
    }
    
    template<typename T>
    bool compare_results(const T* expected, const T* actual, 
                        int count, float tolerance = 1e-5);
};

TEST_F(GFX906KernelTest, TestDot4I8) { /* ... */ }
TEST_F(GFX906KernelTest, TestMatmulQ8) { /* ... */ }
TEST_F(GFX906KernelTest, TestFlashAttention) { /* ... */ }
\`\`\`

## Testing Categories
1. **Correctness**: Bit-exact for INT, tolerance for FP
2. **Performance**: Throughput and latency
3. **Memory**: Bandwidth and access patterns
4. **Edge cases**: Zero sizes, alignment, overflow

## References
- [Testing framework](docs/gfx906/implementation_guide.md#testing-framework)
- Google Test documentation" \
  --label "testing,gfx906" \
  --milestone "Phase 4: Testing & Validation"

# Issue 11: Performance Benchmarking Suite
gh issue create \
  --title "Develop comprehensive performance benchmarking suite" \
  --body "## Description
Create benchmarking tools to measure and track performance improvements.

## Acceptance Criteria
- [ ] Benchmark all optimized kernels
- [ ] Compare against baseline implementation
- [ ] Automated performance regression detection
- [ ] Detailed profiling metrics
- [ ] Performance dashboard/reporting

## Benchmark Components
\`\`\`cpp
struct BenchmarkSuite_gfx906 {
    void benchmark_matmul(int m, int n, int k);
    void benchmark_attention(int seq_len, int head_dim);
    void benchmark_quantization(ggml_type type);
    void measure_memory_bandwidth();
    void profile_kernel_occupancy();
};
\`\`\`

## Key Metrics
- Tokens per second
- TFLOPS achieved
- Memory bandwidth (GB/s)
- Kernel occupancy (%)
- Power efficiency (tokens/watt)

## Profiling Tools
\`\`\`bash
# ROCm profiling
rocprof --stats --timestamp on \\
    --hip-trace --hsa-trace \\
    -o results.csv ./benchmark

# Analysis
rocprof-analyze results.csv
\`\`\`

## References
- [Performance targets](docs/gfx906/optimization_plan.md#performance-targets)
- ROCm profiling documentation" \
  --label "testing,optimization,gfx906" \
  --milestone "Phase 4: Testing & Validation"

# Issue 12: Integration Testing
gh issue create \
  --title "End-to-end integration testing with real models" \
  --body "## Description
Validate optimizations with real-world models and use cases.

## Acceptance Criteria
- [ ] Test with Llama 2 7B, 13B, 70B
- [ ] Test with various quantization levels
- [ ] Perplexity validation
- [ ] Generation quality tests
- [ ] Memory usage validation
- [ ] Multi-batch inference testing

## Test Models
- Llama 2 7B (Q4_0, Q8_0, F16)
- Llama 2 13B (Q4_0, Q5_K_M)
- Mistral 7B
- CodeLlama variants

## Validation Criteria
1. **Accuracy**: Perplexity within 0.1% of reference
2. **Performance**: Meet target speedups
3. **Stability**: 24-hour stress test
4. **Memory**: No leaks, efficient usage

## Test Script
\`\`\`bash
#!/bin/bash
# Integration test suite
for model in llama-7b llama-13b mistral-7b; do
    for quant in q4_0 q8_0 q5_k_m; do
        echo \"Testing $model with $quant\"
        ./llama-bench -m models/$model-$quant.gguf \\
            -p 512 -n 128 -t 1
    done
done
\`\`\`

## References
- [Optimization plan](docs/gfx906/optimization_plan.md)
- Model compatibility matrix" \
  --label "testing,gfx906" \
  --milestone "Phase 4: Testing & Validation"

# Issue 13: Documentation and Examples
gh issue create \
  --title "Create comprehensive documentation and usage examples" \
  --body "## Description
Document all optimizations, APIs, and provide usage examples.

## Acceptance Criteria
- [ ] API documentation for all functions
- [ ] Performance tuning guide
- [ ] Troubleshooting guide
- [ ] Example code for common use cases
- [ ] Migration guide from generic implementation

## Documentation Structure
\`\`\`
docs/gfx906/
├── README.md                    # Overview and quick start
├── optimization_plan.md         # Detailed optimization strategy
├── implementation_guide.md      # Technical implementation
├── docker_setup.md             # Docker environment
├── api_reference.md            # API documentation
├── tuning_guide.md            # Performance tuning
├── troubleshooting.md         # Common issues
└── examples/
    ├── basic_inference.cpp
    ├── batch_processing.cpp
    └── custom_kernel.cpp
\`\`\`

## Example Content
\`\`\`cpp
// Example: Using GFX906 optimized inference
#include \"llama.h\"

int main() {
    // Enable GFX906 optimizations
    llama_backend_init();
    
    // Load model
    auto model = llama_load_model(\"model.gguf\");
    
    // Create context with GFX906 optimizations
    llama_context_params params = llama_context_default_params();
    params.n_gpu_layers = 999;  // Full GPU offload
    
    auto ctx = llama_new_context_with_model(model, params);
    // ...
}
\`\`\`

## References
- Existing llama.cpp documentation
- [Project README](docs/gfx906/README.md)" \
  --label "documentation,gfx906" \
  --milestone "Phase 4: Testing & Validation"

# ============================================================================
# INFRASTRUCTURE AND TOOLING ISSUES
# ============================================================================

echo -e "${GREEN}Infrastructure and Tooling Issues${NC}"

# Issue 14: CI/CD Pipeline
gh issue create \
  --title "Set up CI/CD pipeline for automated testing and benchmarking" \
  --body "## Description
Create automated CI/CD pipeline for continuous testing and performance tracking.

## Acceptance Criteria
- [ ] GitHub Actions workflow for build and test
- [ ] Automated performance regression detection
- [ ] Docker image building and publishing
- [ ] Nightly benchmark runs
- [ ] Results dashboard

## GitHub Actions Workflow
\`\`\`yaml
name: GFX906 CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Nightly

jobs:
  build-and-test:
    runs-on: [self-hosted, gfx906]  # Requires self-hosted runner with GPU
    container:
      image: llama-gfx906:dev
      options: --device=/dev/kfd --device=/dev/dri --group-add video
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Build
        run: |
          cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906
          cmake --build build -j
      
      - name: Test
        run: |
          cd build && ctest -L gfx906
      
      - name: Benchmark
        run: |
          ./build/bin/llama-bench -m test-model.gguf
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: results/
\`\`\`

## References
- GitHub Actions documentation
- Self-hosted runner setup" \
  --label "infrastructure,build,gfx906" \
  --milestone "Phase 1: Foundation"

# Issue 15: Profiling and Analysis Tools
gh issue create \
  --title "Develop profiling and performance analysis tooling" \
  --body "## Description
Create specialized tools for profiling and analyzing GFX906 kernel performance.

## Acceptance Criteria
- [ ] Automated profiling scripts
- [ ] Performance visualization tools
- [ ] Bottleneck analysis
- [ ] Memory usage profiler
- [ ] Power consumption monitoring

## Profiling Script
\`\`\`bash
#!/bin/bash
# profile_gfx906.sh

# Set up environment
export HSA_TOOLS_LIB=/opt/rocm/lib/libroctracer64.so

# Run profiling
rocprof --stats --timestamp on \\
    --hip-trace --hsa-trace \\
    --metric-file gfx906_metrics.txt \\
    -o profile.csv \\
    \"$@\"

# Analyze results
rocprof-analyze profile.csv

# Generate report
python3 scripts/generate_report.py profile.csv
\`\`\`

## Key Metrics
- Memory bandwidth utilization
- Kernel occupancy
- Cache hit rates
- Instruction throughput
- Power consumption

## References
- [Docker setup](docs/gfx906/docker_setup.md#performance-profiling)
- ROCm profiling tools documentation" \
  --label "tooling,optimization,gfx906" \
  --milestone "Phase 4: Testing & Validation"

echo ""
echo -e "${GREEN}✅ Issue creation complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Review created issues on GitHub"
echo "2. Assign team members to issues"
echo "3. Set up project board for tracking"
echo "4. Begin with Phase 1 foundation issues"
echo ""
echo "View all issues:"
echo "  gh issue list --label gfx906"
echo ""
echo "View by milestone:"
echo "  gh issue list --milestone 'Phase 1: Foundation'"
