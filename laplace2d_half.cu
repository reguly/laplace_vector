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

__global__ void stencil(int imax, int jmax, __half *Anew, __half *A) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  if (i < imax+1 && j < jmax+1 && i>0 && j>0) {
    __half val = __hadd(__hadd(A[(j)*(imax+2)+i+1], A[(j)*(imax+2)+i-1]),
                       __hadd(A[(j-1)*(imax+2)+i], A[(j+1)*(imax+2)+i]));
    Anew[(j)*(imax+2)+i] = __hmul(val, __float2half(0.25f));
  }
}

__global__ void copy(int imax, int jmax, __half *Anew, __half *A) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  if (i < imax+1 && j < jmax+1 && i>0 && j>0)
    A[(j)*(imax+2)+i] = Anew[(j)*(imax+2)+i];
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

    __half *A;
    __half *Anew;
    float *y0; // Keep y0 as float for boundary condition calculations

    A    = (__half *)malloc((imax+2) * (jmax+2) * sizeof(__half));
    Anew = (__half *)malloc((imax+2) * (jmax+2) * sizeof(__half));
    y0   = (float *)malloc((imax+2) * sizeof(float));

    // Initialize arrays with zeros
    for (int i = 0; i < (imax+2) * (jmax+2); i++) {
        A[i] = __float2half(0.0f);
    }

    // set boundary conditions
    for (int i = 0; i < imax+2; i++)
      A[(0)*(imax+2)+i]   = __float2half(0.0f);

    for (int i = 0; i < imax+2; i++)
      A[(jmax+1)*(imax+2)+i] = __float2half(0.0f);

    for (int j = 0; j < jmax+2; j++)
    {
        y0[j] = sin(pi * j / (jmax+1));
        A[(j)*(imax+2)+0] = __float2half(y0[j]);
    }

    for (int j = 0; j < imax+2; j++)
    {
        y0[j] = sin(pi * j/ (jmax+1));
        A[(j)*(imax+2)+imax+1] = __float2half(y0[j]*exp(-pi));
    }

    printf("Jacobi relaxation Calculation (Half Precision): %d x %d mesh\n", imax+2, jmax+2);


    
    int iter = 0;

    for (int i = 1; i < imax+2; i++)
       Anew[(0)*(imax+2)+i]   = __float2half(0.0f);

    for (int i = 1; i < imax+2; i++)
       Anew[(jmax+1)*(imax+2)+i] = __float2half(0.0f);

    for (int j = 1; j < jmax+2; j++)
        Anew[(j)*(imax+2)+0]   = __float2half(y0[j]);

    for (int j = 1; j < jmax+2; j++)
        Anew[(j)*(imax+2)+jmax+1] = __float2half(y0[j]*expf(-pi));

    __half *d_A, *d_Anew;
    size_t size = (imax+2)*(jmax+2)*sizeof(__half);
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
        dim3 threads(32,8);
        dim3 blocks((imax+2)/32, (jmax+2)/8);
        stencil<<<blocks, threads>>>(imax, jmax, d_Anew, d_A);

        copy<<<blocks, threads>>>(imax, jmax, d_Anew, d_A);
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