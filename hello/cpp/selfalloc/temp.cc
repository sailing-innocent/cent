#include <iostream>


template <typename T>
class my_allocator
{
public:
    using size_type = size_t;
    using pointer = T*;
    using const_pointer = const T*;
    using value_type = T;

    pointer allocate(size_type n)
    {
        return new T[n];
    }

    void deallocate(pointer p, size_type n)
    {
        delete [] p;
    }
};

#include <vector>

int main()
{
    std::vector<int, my_allocator<int>> a;
    a.resize(100);
}