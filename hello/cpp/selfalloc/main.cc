#include <iostream>
#include <cstdlib>
#include <memory>
#include <vector>

template<class T>
struct MyAllocator {
    using value_type = T;
    MyAllocator() = default;
    template<class _Other>
    constexpr MyAllocator(const MyAllocator<_Other>&) noexcept {}

    T* allocate(size_t size) {
        return static_cast<T*>(malloc(size * sizeof(T)));
    }
    void deallocate(T *ptr, size_t size=0) {
        free(ptr);
    }
};

int main() {
    std::cout << "Hello " << std::endl;
    std::vector<int, MyAllocator<int>> vec(10);
    // std::vector<int, std::allocator<int>> vec(10);
    // vec[0] = 1;
    std::cout << "vec size: " << vec.size() << std::endl;
    return 0;
}