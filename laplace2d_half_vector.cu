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
#include <cuda_fp16.h>

// ============================================================================
// Half Precision Vector Configuration
// ============================================================================
typedef __half2 vec_t;
#define ZERO_VEC __float2half2_rn(0.0f)
#define VEC_ELEMENTS 2


__global__ void stencil_vector_half(int imax, int jmax, vec_t *Anew, vec_t *A) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i < (imax+2)/VEC_ELEMENTS && j < jmax+1 && i > 0 && j > 0) {
        // Load vectors
        vec_t current = A[j * (imax+2)/VEC_ELEMENTS + i];
        vec_t left = A[j * (imax+2)/VEC_ELEMENTS + (i-1)];
        vec_t right = A[j * (imax+2)/VEC_ELEMENTS + (i+1)];
        vec_t up = A[(j-1) * (imax+2)/VEC_ELEMENTS + i];
        vec_t down = A[(j+1) * (imax+2)/VEC_ELEMENTS + i];
        
        // For half2: corrected adjacency using half precision arithmetic
        // For element x: left is left.y, right is current.y
        __half left_x = left.y;     // left.y
        __half right_x = current.y; // current.y
        __half up_x = up.x;          // up.x
        __half down_x = down.x;      // down.x
        
        // For element y: left is current.x, right is right.x
        __half left_y = current.x;   // current.x
        __half right_y = right.x;    // right.x
        __half up_y = up.y;         // up.y
        __half down_y = down.y;     // down.y
        
        // Compute stencil using half precision arithmetic
        __half quarter = __float2half(0.25f);
        
        __half result_x = __hmul(quarter, 
            __hadd(__hadd(left_x, right_x), __hadd(up_x, down_x)));
        __half result_y = __hmul(quarter, 
            __hadd(__hadd(left_y, right_y), __hadd(up_y, down_y)));
        
        // Combine results into half2
        Anew[j * (imax+2)/VEC_ELEMENTS + i] = __halves2half2(result_x, result_y);
    }
}

__global__ void stencil_vector_half2(int imax, int jmax, vec_t *Anew, vec_t *A) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i < (imax+2)/VEC_ELEMENTS && j < jmax+1 && i > 0 && j > 0) {
        // Load vectors
        vec_t current = A[j * (imax+2)/VEC_ELEMENTS + i];
        vec_t left_vec = A[j * (imax+2)/VEC_ELEMENTS + (i-1)];
        vec_t right_vec = A[j * (imax+2)/VEC_ELEMENTS + (i+1)];
        vec_t up = A[(j-1) * (imax+2)/VEC_ELEMENTS + i];
        vec_t down = A[(j+1) * (imax+2)/VEC_ELEMENTS + i];
        
        // Pack half2 variables for adjacency using __halves2half2
        // For element x: left is left_vec.y, right is current.y
        // For element y: left is current.x, right is right_vec.x
        __half2 left = __halves2half2(left_vec.y, current.x);   // (left_x, left_y)
        __half2 right = __halves2half2(current.y, right_vec.x); // (right_x, right_y)
        
        // up and down are straightforward
        // up = (up.x, up.y), down = (down.x, down.y)
        
        // Compute stencil using half2 arithmetic functions
        __half2 quarter = __float2half2_rn(0.25f);
        
        // result = 0.25 * (left + right + up + down)
        __half2 sum1 = __hadd2(left, right);
        __half2 sum2 = __hadd2(up, down);
        __half2 total_sum = __hadd2(sum1, sum2);
        __half2 result = __hmul2(quarter, total_sum);
        
        Anew[j * (imax+2)/VEC_ELEMENTS + i] = result;
    }
}

__global__ void copy_vector_half(int imax, int jmax, vec_t *Anew, vec_t *A) {
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
    const __half tol = __float2half(1.0e-3f); // Adjusted tolerance for half precision
    __half error     = __float2half(1.0f);

    // Use half2 vector operations - ensure imax+2 is divisible by VEC_ELEMENTS
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
        __half* first_element = (__half*)&A[(j)*(imax+2)/VEC_ELEMENTS];
        first_element[0] = __float2half(y0[j]);
    }

    for (int j = 0; j < imax+2; j++) {
        y0[j] = sin(pi * j/ (jmax+1));
        // Set right boundary (last element of each row)
        __half* last_element = (__half*)&A[(j)*(imax+2)/VEC_ELEMENTS + (imax+1)/VEC_ELEMENTS];
        last_element[(imax+1)%VEC_ELEMENTS] = __float2half(y0[j]*exp(-pi));
    }

    printf("Jacobi relaxation Calculation (half2 Vector Loads): %d x %d mesh\n", imax+2, jmax+2);

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
        __half* first_element = (__half*)&Anew[(j)*(imax+2)/VEC_ELEMENTS];
        first_element[0] = __float2half(y0[j]);
    }

    for (int j = 0; j < imax+2; j++) {
        // Set right boundary (last element of each row)
        __half* last_element = (__half*)&Anew[(j)*(imax+2)/VEC_ELEMENTS + (imax+1)/VEC_ELEMENTS];
        last_element[(imax+1)%VEC_ELEMENTS] = __float2half(y0[j]*expf(-pi));
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

    while ( __hge(error, tol) && iter < iter_max )
    {
        error = __float2half(1.0f);
        dim3 threads(32, 8);
        dim3 blocks((imax+2)/VEC_ELEMENTS/32, (jmax+2)/8);
        stencil_vector_half2<<<blocks, threads>>>(imax, jmax, d_Anew, d_A);

        copy_vector_half<<<blocks, threads>>>(imax, jmax, d_Anew, d_A);
        if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, __half2float(error));

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