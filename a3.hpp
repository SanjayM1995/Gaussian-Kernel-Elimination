/*  YOUR_FIRST_NAME
 *  YOUR_LAST_NAME
 *  YOUR_UBIT_NAME
 */

#ifndef A3_HPP
#define A3_HPP

#include <functional>
#include <cuda_runtime_api.h>
#include <iostream>
#include <math.h>

__global__
void run_gaussian_kde(float t,float h,float* x,float* tempArray)
{
  extern __shared__ int sharedData[];
  int threadId = threadIdx.x;
  int index = blockIdx.x *(blockDim.x*2) +threadIdx.x;
  sharedData[threadId] = x[index] +x[index+blockDim.x];
  __syncthreads();

  for(int s = blockDim.x>>1;s>0;s>>=1)
  {
    t = t - sharedData[s];
    __syncthreads();
  }
  t= t/h;
  t = pow(-(t*t)/2);
  t= t*0.56
  if(threadId == 0)
    tempArray[blockIdx.x]  = t;
}

__global__
void final_calc(int n,int h,float *y)
{
  extern __shared__ int sharedData[];
  int threadId = threadId.x;
  int index = blockIdx.x *(blockDim.x*2)+threadIdx.x;
  sharedData[threadId] = tempArray[blockDim.x+index] + tempArray[index];
  for(int s = blockDim.x>>1;s>0;s>>=1)
  {
    if(threadId < s)
    sharedData[threadId] += sharedData[threadId+s];
    __syncthreads();
  }
  sharedData[0] = (n*x -sharedData[0])/(h);

}
__host__
void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) {
    const int block_size = 512;
    int size = n*sizeof(float);
    int num_blocks = (n+block_size - 1) / block_size);
    float* d_x;
    float* d_y;
    float* tempArray;
    cudaMalloc(&d_x,size);
    cudaMalloc(&d_y,size);
    cudaMalloc(&tempArray,size/num_blocks)
    cudaMemcpy(d_x,x.data(),size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,y.data(),size,cudaMemcpyHostToDevice);
    for(int i=0;i<n;i++)
    {
        float temp = x[i];
        run_gaussian_kde<<<num_blocks,block_size>>>(temp,h,&d_x,&tempArray);
        final_calc<<<1,num_blocks>>>(n,h,&d_y);
    }
    cudaMemcpy(d_y,y.data(),size,cudaMemcpyDeviceToHost);
    cudaFree(d_x);
} // gaussian_kde


#endif // A3_HPP
