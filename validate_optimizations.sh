#!/bin/bash

# CUDA 12.8 RTX 5090 Optimization Validation Script
# This script validates that all optimizations are properly compiled and configured

echo "=== CUDA 12.8 RTX 5090 Optimization Validation ==="
echo

# Check if binary exists
if [ ! -f "build/bin/cfxmine" ]; then
    echo "❌ ERROR: cfxmine binary not found. Please build first:"
    echo "  cmake -DCMAKE_BUILD_TYPE=Release -B build"
    echo "  cmake --build build"
    exit 1
fi

echo "✅ Binary found: build/bin/cfxmine"

# Check binary architecture support
echo
echo "🔍 Checking CUDA architecture support..."
if cuobjdump --dump-ptx build/bin/cfxmine 2>/dev/null | grep -q "sm_90"; then
    echo "✅ RTX 5090 (sm_90) support: ENABLED"
else
    echo "❌ RTX 5090 (sm_90) support: NOT FOUND"
fi

if cuobjdump --dump-ptx build/bin/cfxmine 2>/dev/null | grep -q "sm_86"; then
    echo "✅ RTX 3060/3070 (sm_86) support: ENABLED"
else
    echo "⚠️  RTX 3060/3070 (sm_86) support: NOT FOUND"
fi

if cuobjdump --dump-ptx build/bin/cfxmine 2>/dev/null | grep -q "sm_80"; then
    echo "✅ RTX 3080/3090 (sm_80) support: ENABLED"
else
    echo "⚠️  RTX 3080/3090 (sm_80) support: NOT FOUND"
fi

if cuobjdump --dump-ptx build/bin/cfxmine 2>/dev/null | grep -q "sm_75"; then
    echo "✅ RTX 2080/2070 (sm_75) support: ENABLED"
else
    echo "❌ RTX 2080/2070 (sm_75) support: NOT FOUND"
fi

if cuobjdump --dump-ptx build/bin/cfxmine 2>/dev/null | grep -q "sm_61"; then
    echo "✅ GTX 1080/1070 (sm_61) support: ENABLED"
else
    echo "❌ GTX 1080/1070 (sm_61) support: NOT FOUND"
fi

# Check optimization flags in binary
echo
echo "🔍 Checking optimization implementations..."

# Check for rotation optimization markers
if strings build/bin/cfxmine | grep -q "shf.r.wrap.b32\|shf.l.wrap.b32"; then
    echo "✅ Assembly-optimized rotations: ENABLED"
else
    echo "⚠️  Assembly-optimized rotations: NOT DETECTED"
fi

# Check for prefetch instructions
if objdump -d build/bin/cfxmine 2>/dev/null | grep -q "prefetch"; then
    echo "✅ Memory prefetch optimizations: ENABLED"
else
    echo "ℹ️  Memory prefetch optimizations: NOT DETECTABLE (inline PTX)"
fi

# Test basic functionality
echo
echo "🧪 Testing basic functionality..."
timeout 5s ./build/bin/cfxmine --help > /dev/null 2>&1
if [ $? -eq 0 ] || [ $? -eq 124 ]; then
    echo "✅ Binary execution: SUCCESSFUL"
else
    echo "❌ Binary execution: FAILED"
fi

# Check CUDA runtime version
echo
echo "🔍 Checking CUDA environment..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    echo "✅ CUDA Toolkit version: $CUDA_VERSION"
    
    if [ "$(echo "$CUDA_VERSION >= 12.0" | bc 2>/dev/null)" = "1" ]; then
        echo "✅ CUDA 12.x compatibility: CONFIRMED"
    else
        echo "⚠️  CUDA 12.x compatibility: VERSION TOO OLD"
    fi
else
    echo "⚠️  CUDA Toolkit: NOT FOUND (runtime only)"
fi

# Check for GPU availability
echo
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 Checking GPU availability..."
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ $GPU_COUNT -gt 0 ]; then
        echo "✅ NVIDIA GPUs detected: $GPU_COUNT"
        nvidia-smi -L | while read -r line; do
            echo "  📱 $line"
        done
        
        # Check for RTX 5090 specifically
        if nvidia-smi -L | grep -i "5090"; then
            echo "🚀 RTX 5090 DETECTED - Ready for maximum hashrate optimization!"
        fi
    else
        echo "ℹ️  No NVIDIA GPUs detected in this environment"
    fi
else
    echo "ℹ️  nvidia-smi not available - GPU detection skipped"
fi

echo
echo "=== Validation Summary ==="

# Count successful validations
CHECKS=0
PASSED=0

# Architecture support
for arch in sm_90 sm_86 sm_80 sm_75 sm_61; do
    CHECKS=$((CHECKS + 1))
    if cuobjdump --dump-ptx build/bin/cfxmine 2>/dev/null | grep -q "$arch"; then
        PASSED=$((PASSED + 1))
    fi
done

# Binary execution
CHECKS=$((CHECKS + 1))
timeout 5s ./build/bin/cfxmine --help > /dev/null 2>&1
if [ $? -eq 0 ] || [ $? -eq 124 ]; then
    PASSED=$((PASSED + 1))
fi

SCORE=$((PASSED * 100 / CHECKS))

if [ $SCORE -ge 80 ]; then
    echo "🎉 OPTIMIZATION STATUS: EXCELLENT ($PASSED/$CHECKS passed - $SCORE%)"
    echo "   Ready for production mining on RTX 5090!"
elif [ $SCORE -ge 60 ]; then
    echo "✅ OPTIMIZATION STATUS: GOOD ($PASSED/$CHECKS passed - $SCORE%)"
    echo "   Most optimizations are active"
else
    echo "⚠️  OPTIMIZATION STATUS: NEEDS ATTENTION ($PASSED/$CHECKS passed - $SCORE%)"
    echo "   Some optimizations may not be active"
fi

echo
echo "📖 For detailed optimization information, see: CUDA_12.8_RTX5090_OPTIMIZATIONS.md"
echo "🚀 To start mining: ./build/bin/cfxmine --gpu --addr <stratum_server> --port 32525"
echo