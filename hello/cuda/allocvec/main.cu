#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

// eastl

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

__global__ void kernel(int *arr, int n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        arr[i] = i;
    }
}

int main() {
    int n = 10;
    std::vector<int, CudaAllocator<int>> arr(n);
    kernel<<<2, 8>>>(arr.data(), n);
    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++) {
        printf("arr[%d]: %d\n", i, arr[i]);
    }

    return 0;
}