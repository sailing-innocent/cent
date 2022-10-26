#include <iostream>
#define N 10
extern "C" {
    __global__ void add(int *a, int *b, int *c);
    __declspec(dllexport) void printHello();
}

