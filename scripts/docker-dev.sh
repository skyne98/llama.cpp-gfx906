#!/bin/bash
# Docker development environment setup for GFX906

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Setting up GFX906 Docker Development Environment${NC}"

# Check for GPU
if ! lspci | grep -q "AMD.*Vega 20"; then
    echo -e "${YELLOW}⚠️  Warning: AMD Vega 20 (gfx906) GPU not detected${NC}"
    echo "Detected GPUs:"
    lspci | grep -E "(VGA|3D|Display)" || echo "No GPUs found"
fi

# Check ROCm installation on host
if ! command -v rocminfo &> /dev/null; then
    echo -e "${YELLOW}⚠️  ROCm not found on host. Docker will use containerized ROCm.${NC}"
else
    echo -e "${GREEN}✓ ROCm found on host${NC}"
    rocminfo | grep gfx906 || echo -e "${YELLOW}Note: gfx906 not detected by rocminfo${NC}"
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check docker-compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose not installed. Please install Docker Compose.${NC}"
    exit 1
fi

# Build development image
echo -e "${GREEN}Building development Docker image...${NC}"
docker compose build gfx906-dev

# Create necessary directories
mkdir -p models benchmarks

# Start development container
echo -e "${GREEN}Starting development container...${NC}"
docker compose run --rm \
    --name gfx906-dev \
    gfx906-dev \
    /bin/bash -c '
        echo -e "${GREEN}==================================${NC}"
        echo -e "${GREEN}  GFX906 Development Environment  ${NC}"
        echo -e "${GREEN}==================================${NC}"
        echo ""
        echo "Checking GPU access..."
        if rocminfo | grep -q gfx906; then
            echo -e "${GREEN}✓ GFX906 GPU detected!${NC}"
            rocm-smi --showproductname
        else
            echo -e "${YELLOW}⚠️  GFX906 not detected. Check HSA_OVERRIDE_GFX_VERSION${NC}"
        fi
        echo ""
        echo "Available commands:"
        echo "  rocm-smi          - Monitor GPU"
        echo "  rocminfo          - GPU information"
        echo "  cmake             - Build system"
        echo "  ninja             - Fast build tool"
        echo "  rocprof           - Profiling tool"
        echo ""
        echo "Project location: /workspace/llama.cpp-gfx906"
        echo ""
        exec /bin/bash
    '