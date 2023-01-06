#include <iostream>
#include <cuda_runtime.h>

// __inline__ // __force_inline__ // __no_inline__
__device__ void say_hello() {
    printf("Hello From GPU!");
}

__global__ void kernel(int *pret) {
    printf("Block [%d,%d,%d] of [%d,%d,%d], Thread [%d,%d,%d] of [%d,%d,%d]\n", blockIdx.x, blockIdx.y,  blockIdx.z, gridDim.x, gridDim.y, gridDim.z, 
        threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y, blockDim.z);
    // say_hello();
    *pret = 1;
}
// kernel could also execute kernel

__host__ void say_hello_CPU() {
    printf("Hello From CPU!");
}

// __host__ __device__ will generate 2 version

constexpr const char* cuthead(const char *p) {
    return p+1;
}
// constexpr will be compiled both in GPU and CPU 
// enable --expt-relaxed-constexpr, we can set it only able for .cu file // otherwise gcc will export error
// cannot printf 
// could use __CUDA_ARCH__
// __CUDA_ARCH__ 520 // 52 系列显卡
// set(CMAKE_CUDA_ARCHITECTURE 52;75;86) 但是不会报错，编译会变慢，可执行文件也会变大

int main() {
    dim3 BLOCK_DIM = dim3(1,1,1);
    dim3 GRID_DIM = dim3(1,1,1);
    int* pret;
    cudaMalloc(&pret, sizeof(int));
    kernel<<<GRID_DIM,BLOCK_DIM>>>(pret); // synchroize, will not return on time
    // cudaError_t err = cudaDeviceSynchronize(); // wait until all of the gpu idle
    // cudaMallocManaged for Pascal version -- not sync
    int ret;
    cudaMemcpy(&ret, pret, sizeof(int), cudaMemcpyDeviceToHost); // auto synchronize
    printf("result: %d \n", ret);
    cudaFree(pret);
    // say_hello_CPU();
    return 0;
}