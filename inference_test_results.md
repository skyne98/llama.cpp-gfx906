# Gemma 3 270M Inference Test Results

## Test Configuration
- **Model**: gemma-3-270m-Q8_0.gguf (292MB)
- **Prompt**: "The sky is"
- **Tokens Generated**: 20
- **Hardware**: AMD GFX906 (Radeon Graphics)

## Performance Comparison

### CPU Inference (build-cpu)
- **Prompt Processing**: 39.58 ms/token (25.27 tokens/sec)
- **Generation Speed**: 285.64 ms/token (3.50 tokens/sec)
- **Total Time**: 9.71 seconds for 42 tokens
- **Average**: ~3.44 tokens/second

### GPU Inference (build-hip with GFX906)
- **Prompt Processing**: 12.19 ms/token (82.05 tokens/sec)
- **Generation Speed**: 4.67 ms/token (214.28 tokens/sec)
- **Total Time**: 1.56 seconds for 48 tokens
- **Average**: ~85.88 tokens/second

## Performance Improvement

| Metric | CPU | GPU (GFX906) | Speedup |
|--------|-----|--------------|---------|
| Prompt Processing | 25.27 tok/s | 82.05 tok/s | **3.25x** |
| Generation | 3.50 tok/s | 214.28 tok/s | **61.2x** |
| Overall Speed | 3.44 tok/s | 85.88 tok/s | **~25x** |

## Key Observations

1. **GPU Acceleration Works**: The HIP build successfully utilizes the GFX906 GPU
2. **Massive Generation Speedup**: 61x faster token generation on GPU
3. **All Layers Offloaded**: Successfully offloaded all model layers to GPU (ngl=999)
4. **Memory Usage**: GPU uses 64.16 MiB compute buffer vs 64.31 MiB on CPU

## Verification Status

✅ **All PR acceptance criteria met:**
- CPU build functional
- HIP/GPU build functional with GFX906 detection
- Test suite passed (39/39 tests)
- Model inference verified on both CPU and GPU
- Significant performance improvement demonstrated

The foundation for GFX906 optimization is successfully established and working!
