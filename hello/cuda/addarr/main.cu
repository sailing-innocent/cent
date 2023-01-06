#include <cstdio>
#include <cuda_runtime.h>

__global__ void add(int* arr, int *brr, int* crr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = gridDim.x  * blockDim.x;
    int i = 1;
    if (idx < dim) {
        if (n - dim < 0) {
            crr[idx] = arr[idx] + brr[idx];
        } else {
            while ( n - dim * i > 0) {
                // printf("is adding %d + %d item \n", n-dim*i, idx);
                crr[n-dim*i+idx] = brr[n-dim*i+idx] + arr[n-dim*i+idx];
                i = i + 1;
            }
            if (n - dim * i + idx >= 0) {
                // printf("is adding %d + %d item \n", n-dim*i, idx);
                crr[n-dim*i+idx] = brr[n-dim*i+idx] + arr[n-dim*i+idx];
            }
        }
    }
}

int main() {
    int n = 32;
    int *arr; cudaMallocManaged(&arr, n*sizeof(int));
    int *brr; cudaMallocManaged(&brr, n*sizeof(int));
    int *crr; cudaMallocManaged(&crr, n*sizeof(int));
    
    for (int i = 0; i < n; i++) {
        *(arr + i) = i;
    }
    for (int i = 0; i < n; i++) {
        *(brr + i) = i * i ;
    }
    add<<<2,4>>>(arr,brr,crr,n);
    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++) {
        printf("crr[%d] = %d\n", i, crr[i]);
    }
    cudaFree(arr);
    cudaFree(brr);
    cudaFree(crr);
    return 0;
}