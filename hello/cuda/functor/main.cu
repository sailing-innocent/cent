#include <iostream>
#include <cuda_runtime.h>

template<class Func>
__global__ void parallel_for(int n, Func func) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
        i < n; i += blockDim.x * gridDim.x ) {
            func(i);
    }
}

struct MyFunctor {
    __device__ void operator()(int i) const {
        printf("Number %d \n", i);
    }
};

int main() {
    int n = 10;
    parallel_for<<<2,8>>>(n, MyFunctor{});
    cudaDeviceSynchronize();
    return 0;
}