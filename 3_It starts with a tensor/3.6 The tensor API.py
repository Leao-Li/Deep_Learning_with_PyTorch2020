import torch

a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
print(a)
print(a_t)

a.shape, a_t.shape

a = torch.ones(3, 2)
a_t = a.transpose(0, 1)

a.shape, a_t.shape

# We mentioned the online docs earlier (http://pytorch.org/docs). 

"""
Creation ops 创建张量操作

Indexing, slicing, joining, mutating ops 索引， 分割，连接，变换操作 

Math ops: 数学操作
    -Pointwise ops 点态运算  
     Functions for obtaining a new tensor by applying a function to
     each element independtly, like abs and cos
    -Reduction ops 降维
     Functions for computing aggregate values by iterating
     through tensors, like mean, std, and norm
    -Comparison ops 比较
     Functions for evaluating numerical predicates over tensors,
     and like equal and max
    -Spectral ops
     Functions for transforming in and operating in the frequency
     domain, like stft and hamming_window
    -Other operations 
     Special functions operating on vectors, like cross, or matrices,
     like trace
    -BLAS and LAPACK operations 线性代数操作
     Functions following the Basic Linear Algebra Subprogram (BLAS)
     specification for scalar, vector-vector, matrix-vector, and
     matrix-matrix operations

Random sampling - Functions for generating values by drawing randomly
    from probability distribution, like randn and normal

Serialization - Functions for saving and loading tensors, like load and save

Parallelism - Functions for controlling the number of threads for parallel
    CPU execution, like set_num_threads
"""