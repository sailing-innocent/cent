#include <iostream>
#include <cuda_runtime.h>

template <class T>
struct CudaAllocator {
    using value_type = T;
    CudaAllocator() = default;
    template<class _Other>
    constexpr CudaAllocator(const CudaAllocator<_Other>&) noexcept {}

    T *allocate(size_t size) {
        T *ptr = nullptr;
        cudaMallocManaged(&ptr, size * sizeof(T));
        return ptr;
    }

    void deallocate(T *ptr, size_t size = 0) {
        cudaFree(ptr);
    }

    // plain old data
    template <class ...Args>
    void constructor(T *p, Args &&...args) {
        if constexpr (!(sizeof...(Args) == 0 && std::is_pod_v<T>))
        {
            ::new((void*)p) T(std::forward<Args>(args)...);
        }
    }
};

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
    // parallel_for<<<2,8>>>(n, MyFunctor{});
    parallel_for<<<2,8>>>(n, [] __device__ (int i) {
        printf("lambda Number %d \n", i);
    });
    cudaDeviceSynchronize();
    return 0;
}