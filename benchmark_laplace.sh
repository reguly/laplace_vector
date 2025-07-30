#!/bin/bash

# Benchmark script for Laplace 2D solver variants
# Compiles and profiles different precision and vectorization versions

set -e  # Exit on any error

echo "=========================================="
echo "Laplace 2D Solver Benchmark Suite"
echo "=========================================="

# Clean up any existing binaries
rm -f laplace2d laplace2d_half laplace2d_vector laplace2d_half_vector
rm -f *.nsys-rep *.qdrep

# Compilation flags
NVCC_FLAGS="-O3 -arch=sm_80"  # Adjust compute capability as needed
NVCC_FP16_FLAGS="-O3 -arch=sm_80"  # sm_53+ required for half precision

echo "Compiling versions..."

# Compile single precision version
echo "  - Compiling laplace2d.cu (single precision)..."
nvcc $NVCC_FLAGS -o laplace2d laplace2d.cu

# Compile half precision version
echo "  - Compiling laplace2d_half.cu (half precision)..."
nvcc $NVCC_FP16_FLAGS -o laplace2d_half laplace2d_half.cu

# Compile vectorized float version (currently set to float4)
echo "  - Compiling laplace2d_vector.cu (vectorized float)..."
nvcc $NVCC_FLAGS -o laplace2d_vector laplace2d_vector.cu

# Compile vectorized half precision version
echo "  - Compiling laplace2d_half_vector.cu (vectorized half precision)..."
nvcc $NVCC_FP16_FLAGS -o laplace2d_half_vector laplace2d_half_vector.cu

echo "Compilation complete!"
echo ""

# Function to extract kernel stats from nsys output
extract_kernel_stats() {
    local output_file="$1"
    local version_name="$2"
    
    echo "----------------------------------------"
    echo "$version_name Kernel Statistics:"
    echo "----------------------------------------"
    
    # Extract the 6 lines after "Executing 'cuda_gpu_kern_sum' stats report"
    grep -A 6 "Executing 'cuda_gpu_kern_sum' stats report" "$output_file" | tail -6
    echo ""
}

# Function to run nsys profiling
run_profile() {
    local binary="$1"
    local version_name="$2"
    local output_prefix="$3"
    
    echo "Profiling $version_name..."
    
    # Run nsys with stats output
    nsys profile \
        --stats=true \
        --force-overwrite=true \
        --output="$output_prefix" \
        ./"$binary" > "${output_prefix}_output.txt" 2>&1
    
    # Extract kernel statistics
    extract_kernel_stats "${output_prefix}_output.txt" "$version_name"
}

echo "Running profiling benchmarks..."
echo ""

# Profile each version
run_profile "laplace2d" "Single Precision" "laplace2d_profile"
run_profile "laplace2d_half" "Half Precision" "laplace2d_half_profile"
run_profile "laplace2d_vector" "Vectorized Float" "laplace2d_vector_profile"
run_profile "laplace2d_half_vector" "Vectorized Half Precision" "laplace2d_half_vector_profile"
