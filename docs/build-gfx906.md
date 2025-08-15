# Building llama.cpp for AMD Instinct MI50 (GFX906)

This guide provides specific instructions for building llama.cpp with optimizations for AMD Instinct MI50 GPUs (gfx906 architecture).

## Prerequisites

1. **ROCm Installation** (5.7+ recommended, 6.x supported)
   ```bash
   # Ubuntu/Debian
   wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
   echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
   sudo apt update
   sudo apt install rocm-dev hipblas rocblas
   
   # Add user to video/render groups
   sudo usermod -a -G video,render $USER
   # Logout and login for group changes to take effect
   ```

2. **Build Tools**
   ```bash
   sudo apt install cmake build-essential git
   ```

## Quick Build Instructions

```bash
# Clone the repository
git clone https://github.com/skyne98/llama.cpp-gfx906.git
cd llama.cpp-gfx906

# CRITICAL: Initialize the ggml-gfx906 submodule
git submodule update --init --recursive

# Build with GFX906 optimizations
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)
```

## Verify GPU Detection

After building, verify that your MI50 is properly detected:

```bash
# Check ROCm detection
rocm-smi

# Test with a model
./build/bin/llama-cli -m path/to/model.gguf -p "Hello" -n 10
```

## Performance Optimizations

The ggml-gfx906 fork includes specific optimizations for MI50:

### Hardware Instructions Used
- **V_DOT4_I32_I8**: 4x INT8 dot product operations
- **V_DOT2_F32_F16**: 2x FP16 dot product operations
- **V_PK_FMA_F16**: Dual FP16 FMA operations
- **DS_PERMUTE/BPERMUTE**: Hardware lane shuffling

### Expected Performance Improvements
- Q8_0 quantization: ~40% improvement over baseline
- Q4_0 quantization: ~55% improvement over baseline
- Flash Attention: ~35% improvement
- Memory bandwidth: Up to 900 GB/s (HBM2)

## Docker Build

For consistent builds, use the provided Docker configuration:

```bash
# Build Docker image
docker build -f Dockerfile.gfx906 -t llama-gfx906 .

# Run with GPU support
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  -v ./models:/models \
  llama-gfx906 \
  ./bin/llama-cli -m /models/your-model.gguf -p "Test" -n 20
```

## Troubleshooting

### Missing ggml files during build
```bash
# Ensure submodule is initialized
git submodule update --init --recursive
```

### GPU not detected
```bash
# Check GPU visibility
rocm-smi
export HIP_VISIBLE_DEVICES=0  # Use first GPU
```

### Build errors with HIP
```bash
# Set explicit paths if needed
export HIPCXX="$(hipconfig -l)/clang"
export HIP_PATH="$(hipconfig -R)"
```

## Development

The GFX906 optimizations are implemented in the [ggml-gfx906 fork](https://github.com/skyne98/ggml-gfx906). To contribute:

1. Work on optimizations in the ggml fork
2. Test changes locally
3. Update the submodule reference in llama.cpp

See the [ggml-gfx906 issues](https://github.com/skyne98/ggml-gfx906/issues) for ongoing optimization work.

## Related Documentation

- [Main build documentation](./build.md)
- [Docker documentation](./docker.md)
- [GFX906 optimization plan](../docs/gfx906/optimization_plan.md)
- [Implementation guide](../docs/gfx906/implementation_guide.md)