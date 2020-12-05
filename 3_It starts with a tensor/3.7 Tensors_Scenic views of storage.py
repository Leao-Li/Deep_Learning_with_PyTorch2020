# 3.7 Tensors: Scenic views of storage
# 张量在内存中的存储形态
import torch

## 3.7.1 Indexing into storage
# Create 2D tensor
points = torch.tensor([[10.0, 9.0], [5.0, 3.0], [2.0, 1.0]])
points.storage()  # 二维张量在内存的形态

points_storage = points.storage()
points_storage[0]
points_storage[-1]
points_storage[-2]

points.storage()[1]
points.storage()[-1]


points = torch.tensor([[2.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
points_storage = points.storage()
points_storage[0] = 100.0
points


## 3.7.2 Modifying stored values: In-place operations
a = torch.ones(2, 3)
print(a)

a.zero_() # change the value of variabl a in storage 
print(a)


