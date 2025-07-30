/* Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================================
// Vector Configuration - Change this to switch between float2 and float4
// ============================================================================
#define VECTOR_SIZE 4  // Change to 4 for float4 version

#if VECTOR_SIZE == 2
    typedef float2 vec_t;
    #define MAKE_VEC(x, y) make_float2(x, y)
    #define ZERO_VEC make_float2(0.0f, 0.0f)
    #define VEC_ELEMENTS 2
#elif VECTOR_SIZE == 4
    typedef float4 vec_t;
    #define MAKE_VEC(x, y, z, w) make_float4(x, y, z, w)
    #define ZERO_VEC make_float4(0.0f, 0.0f, 0.0f, 0.0f)
    #define VEC_ELEMENTS 4
#else
    #error "VECTOR_SIZE must be 2 or 4"
#endif

__global__ void stencil_vector(int imax, int jmax, vec_t *Anew, vec_t *A) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i < (imax+2)/VEC_ELEMENTS && j < jmax+1 && i > 0 && j > 0) {
        // Load vectors
        vec_t current = A[j * (imax+2)/VEC_ELEMENTS + i];
        vec_t left = A[j * (imax+2)/VEC_ELEMENTS + (i-1)];
        vec_t right = A[j * (imax+2)/VEC_ELEMENTS + (i+1)];
        vec_t up = A[(j-1) * (imax+2)/VEC_ELEMENTS + i];
        vec_t down = A[(j+1) * (imax+2)/VEC_ELEMENTS + i];
        
        vec_t result;
        
#if VECTOR_SIZE == 2
        // For float2: corrected adjacency
        // For element x: left is left.y, right is current.y
        result.x = 0.25f * (left.y + current.y + up.x + down.x);
        // For element y: left is current.x, right is right.x
        result.y = 0.25f * (current.x + right.x + up.y + down.y);
        
#elif VECTOR_SIZE == 4
        // For float4: corrected adjacency for all 4 elements
        // For element 0: left is left_vec.w, right is current.y
        result.x = 0.25f * (left.w + current.y + up.x + down.x);
        // For element 1: left is current.x, right is current.z
        result.y = 0.25f * (current.x + current.z + up.y + down.y);
        // For element 2: left is current.y, right is current.w
        result.z = 0.25f * (current.y + current.w + up.z + down.z);
        // For element 3: left is current.z, right is right_vec.x
        result.w = 0.25f * (current.z + right.x + up.w + down.w);
#endif
        
        Anew[j * (imax+2)/VEC_ELEMENTS + i] = result;
    }
}

__global__ void copy_vector(int imax, int jmax, vec_t *Anew, vec_t *A) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i < (imax+2)/VEC_ELEMENTS && j < jmax+1 && i > 0 && j > 0) {
        A[j * (imax+2)/VEC_ELEMENTS + i] = Anew[j * (imax+2)/VEC_ELEMENTS + i];
    }
}

int main(int argc, char** argv)
{
    //Size along y
    int jmax = 8190;
    //Size along x
    int imax = 8190;
    int iter_max = 2;

    const double pi  = 2.0 * asin(1.0);
    const float tol = 1.0e-5f;
    float error     = 1.0f;

    // Use vector operations - ensure imax+2 is divisible by VEC_ELEMENTS
    if ((imax + 2) % VEC_ELEMENTS != 0) {
        imax = ((imax + 2) / VEC_ELEMENTS) * VEC_ELEMENTS - 2;
        printf("Adjusted imax to %d to ensure divisibility by %d\n", imax, VEC_ELEMENTS);
    }

    vec_t *A;
    vec_t *Anew;
    float *y0; // Keep y0 as float for boundary condition calculations

    A    = (vec_t *)malloc((imax+2)/VEC_ELEMENTS * (jmax+2) * sizeof(vec_t));
    Anew = (vec_t *)malloc((imax+2)/VEC_ELEMENTS * (jmax+2) * sizeof(vec_t));
    y0   = (float *)malloc((imax+2) * sizeof(float));

    // Initialize arrays with zeros
    memset(A, 0, (imax+2)/VEC_ELEMENTS * (jmax+2) * sizeof(vec_t));

    // set boundary conditions
    for (int i = 0; i < (imax+2)/VEC_ELEMENTS; i++) {
        A[(0)*(imax+2)/VEC_ELEMENTS+i] = ZERO_VEC;
    }

    for (int i = 0; i < (imax+2)/VEC_ELEMENTS; i++) {
        A[(jmax+1)*(imax+2)/VEC_ELEMENTS+i] = ZERO_VEC;
    }

    for (int j = 0; j < jmax+2; j++) {
        y0[j] = sin(pi * j / (jmax+1));
        // Set left boundary (first element of each row)
        float* first_element = (float*)&A[(j)*(imax+2)/VEC_ELEMENTS];
        first_element[0] = y0[j];
    }

    for (int j = 0; j < imax+2; j++) {
        y0[j] = sin(pi * j/ (jmax+1));
        // Set right boundary (last element of each row)
        float* last_element = (float*)&A[(j)*(imax+2)/VEC_ELEMENTS + (imax+1)/VEC_ELEMENTS];
        last_element[(imax+1)%VEC_ELEMENTS] = y0[j]*exp(-pi);
    }

    printf("Jacobi relaxation Calculation (float%d Vector Loads): %d x %d mesh\n", VEC_ELEMENTS, imax+2, jmax+2);

    int iter = 0;

    // Initialize Anew boundary conditions
    for (int i = 0; i < (imax+2)/VEC_ELEMENTS; i++) {
        Anew[(0)*(imax+2)/VEC_ELEMENTS+i] = ZERO_VEC;
    }

    for (int i = 0; i < (imax+2)/VEC_ELEMENTS; i++) {
        Anew[(jmax+1)*(imax+2)/VEC_ELEMENTS+i] = ZERO_VEC;
    }

    for (int j = 0; j < jmax+2; j++) {
        // Set left boundary (first element of each row)
        float* first_element = (float*)&Anew[(j)*(imax+2)/VEC_ELEMENTS];
        first_element[0] = y0[j];
    }

    for (int j = 0; j < imax+2; j++) {
        // Set right boundary (last element of each row)
        float* last_element = (float*)&Anew[(j)*(imax+2)/VEC_ELEMENTS + (imax+1)/VEC_ELEMENTS];
        last_element[(imax+1)%VEC_ELEMENTS] = y0[j]*expf(-pi);
    }

    vec_t *d_A, *d_Anew;
    size_t size = (imax+2)/VEC_ELEMENTS*(jmax+2)*sizeof(vec_t);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_Anew, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Anew, Anew, size, cudaMemcpyHostToDevice);
    
    // CUDA timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    while ( error > tol && iter < iter_max )
    {
        error = 1.0f;
        dim3 threads(32, 8);
        dim3 blocks((imax+2)/VEC_ELEMENTS/32, (jmax+2)/8);
        stencil_vector<<<blocks, threads>>>(imax, jmax, d_Anew, d_A);

        copy_vector<<<blocks, threads>>>(imax, jmax, d_Anew, d_A);
        if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);

        iter++;
    }

    // Stop timing and calculate runtime
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float runtime;
    cudaEventElapsedTime(&runtime, start, stop);
    runtime /= 1000.0f; // Convert from milliseconds to seconds

    printf(" total: %f s\n", runtime);
    
    // Clean up CUDA events and memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_Anew);
    free(A);
    free(Anew);
    free(y0);
} 