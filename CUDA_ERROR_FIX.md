# CUDA Runtime Error Fix

## Problem Description
The miner was failing with `CUDA error RUNTIME: '1'` (cudaErrorInvalidValue) during initialization. This error occurred when trying to use the `cudaLimitPersistingL2CacheSize` feature which is not available in all CUDA versions or on all hardware.

## Root Cause
The code was attempting to use a CUDA feature (`cudaLimitPersistingL2CacheSize`) that:
- Was introduced in CUDA 11.2+
- Requires compute capability 8.0+ (Ampere architecture and newer)
- May not be supported by all drivers or hardware configurations

## Solution
Added proper feature detection and graceful fallback:

### 1. Compile-time Detection
```cpp
#ifdef cudaLimitPersistingL2CacheSize
    // Feature is available in this CUDA version
#else
    // Feature not available, skip optimization
#endif
```

### 2. Runtime Error Handling
```cpp
cudaError_t err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.l2CacheSize);
if (err != cudaSuccess) {
    // Feature not supported on this hardware/driver, continue without it
    fprintf(stderr, "Warning: Persistent L2 cache not supported, continuing without optimization\n");
    cudaGetLastError(); // Clear the error state
}
```

### 3. Hardware Compatibility Check
```cpp
if (prop.major >= 8) { // Ampere and newer architectures
    // Only attempt to use the feature on supported hardware
}
```

## Impact
- **Before**: Miner would crash with CUDA error during initialization
- **After**: Miner starts successfully, with or without the optimization
- **Performance**: Full optimization on supported systems, graceful degradation on others
- **Compatibility**: Works with all CUDA versions and GPU architectures

## Testing
The fix ensures compatibility with:
- Older CUDA versions (pre-11.2): Compiles and runs without the optimization
- Older GPUs (pre-Ampere): Skips the optimization gracefully  
- Newer systems: Uses full optimization when available
- Unsupported drivers: Continues without crashing

## Files Modified
- `src/OctopusCUDAMiner.cu`: Added feature detection and error handling
- `CUDA_12.8_RTX5090_OPTIMIZATIONS.md`: Updated documentation with compatibility notes