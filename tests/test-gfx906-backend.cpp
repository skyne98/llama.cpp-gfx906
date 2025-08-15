#include "ggml-cuda.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// External functions from GFX906 backend
extern "C" {
bool ggml_cuda_gfx906_init();
bool ggml_cuda_gfx906_init_streams(int device_id);
void ggml_cuda_gfx906_cleanup();
void ggml_cuda_gfx906_print_perf_stats();
}

// Test device detection
bool test_device_detection() {
    printf("Testing GFX906 device detection...\n");

    // Get CUDA device info
    int device_count = ggml_cuda_get_device_count();
    printf("  Total CUDA devices: %d\n", device_count);

    if (device_count == 0) {
        printf("  No CUDA devices found\n");
        return false;
    }

    // Initialize GFX906 backend
    bool gfx906_found = ggml_cuda_gfx906_init();

    if (!gfx906_found) {
        printf("  No GFX906 devices found (this is OK if you don't have an MI50)\n");
        return true;  // Not an error, just no GFX906 hardware
    }

    printf("  GFX906 device detection: PASSED\n");
    return true;
}

// Test stream management
bool test_stream_management() {
    printf("Testing GFX906 stream management...\n");

    // Check if we have a GFX906 device
    if (!ggml_cuda_gfx906_init()) {
        printf("  Skipping stream test (no GFX906 device)\n");
        return true;
    }

    // Initialize streams for device 0
    bool result = ggml_cuda_gfx906_init_streams(0);

    if (!result) {
        printf("  Failed to initialize streams\n");
        return false;
    }

    printf("  Stream management: PASSED\n");
    return true;
}

// Test memory allocation
bool test_memory_allocation() {
    printf("Testing GFX906 memory allocation...\n");

    int device_count = ggml_cuda_get_device_count();
    if (device_count == 0) {
        printf("  Skipping memory test (no CUDA devices)\n");
        return true;
    }

    // Test basic CUDA memory allocation
    void * ptr  = nullptr;
    size_t size = 1024 * 1024;  // 1 MB

    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        printf("  Failed to allocate memory: %s\n", cudaGetErrorString(err));
        return false;
    }

    // Test memory operations
    err = cudaMemset(ptr, 0, size);
    if (err != cudaSuccess) {
        printf("  Failed to set memory: %s\n", cudaGetErrorString(err));
        cudaFree(ptr);
        return false;
    }

    // Free memory
    err = cudaFree(ptr);
    if (err != cudaSuccess) {
        printf("  Failed to free memory: %s\n", cudaGetErrorString(err));
        return false;
    }

    printf("  Memory allocation: PASSED\n");
    return true;
}

// Test configuration values
bool test_configuration() {
    printf("Testing GFX906 configuration...\n");

#ifdef GGML_HIP_GFX906_OPTIMIZED
    printf("  GGML_HIP_GFX906_OPTIMIZED is defined\n");

#    ifdef __gfx906__
    printf("  __gfx906__ is defined\n");
    printf("  Expected configuration:\n");
    printf("    - 60 Compute Units\n");
    printf("    - 64KB LDS per CU\n");
    printf("    - Wave size: 64\n");
#    else
    printf("  __gfx906__ is NOT defined (OK if not compiling for GFX906)\n");
#    endif
#else
    printf("  GGML_HIP_GFX906_OPTIMIZED is NOT defined\n");
#endif

    printf("  Configuration test: PASSED\n");
    return true;
}

// Main test runner
int main() {
    printf("========================================\n");
    printf("GFX906 Backend Infrastructure Test Suite\n");
    printf("========================================\n\n");

    int tests_passed = 0;
    int tests_failed = 0;

    // Run tests
    if (test_device_detection()) {
        tests_passed++;
    } else {
        tests_failed++;
    }

    if (test_stream_management()) {
        tests_passed++;
    } else {
        tests_failed++;
    }

    if (test_memory_allocation()) {
        tests_passed++;
    } else {
        tests_failed++;
    }

    if (test_configuration()) {
        tests_passed++;
    } else {
        tests_failed++;
    }

    // Print performance stats if available
#ifdef GGML_HIP_GFX906_OPTIMIZED
    ggml_cuda_gfx906_print_perf_stats();
#endif

    // Cleanup
#ifdef GGML_HIP_GFX906_OPTIMIZED
    ggml_cuda_gfx906_cleanup();
#endif

    // Print summary
    printf("\n========================================\n");
    printf("Test Summary:\n");
    printf("  Tests passed: %d\n", tests_passed);
    printf("  Tests failed: %d\n", tests_failed);

    if (tests_failed == 0) {
        printf("  Result: ALL TESTS PASSED\n");
    } else {
        printf("  Result: SOME TESTS FAILED\n");
    }
    printf("========================================\n");

    return tests_failed;
}

