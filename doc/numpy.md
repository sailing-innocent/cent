# Numpy

Numpy's array is called `ndarray`. It is also known by the alias array.

- ndarray.ndim: The number of axis
- ndarray.shape: The length of shape tuple is ndim
- ndarray.size: total number of element
- ndarray.dtype: the data type of array
- ndarray.itemsize: the size in bytes of each element of the array
- ndarary.data: the buffer containing actual elements of array

用c++的思维来看numpy的结构其实更容易一些，里层相当于一个buffer，外面这些只是不同的描述结构。所以可以比较方便地reshape，毕竟只是更改些外层接口参数而已，不会动底层存储结构，就会比较快捷。

