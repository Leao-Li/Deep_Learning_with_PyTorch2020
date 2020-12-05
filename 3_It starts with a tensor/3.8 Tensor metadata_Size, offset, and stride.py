# 3.8 Tensor metadata: Size, offset, and stride

import torch

## 3.8.1 Views of another tensor’s storage
points = torch.tensor([[40.0, 10.0], [5.0, 3.0], [2.0, 1.0]])
print(points)

second_point = points[1]
second_point.storage_offset()

second_point.size()

second_point.shape

points.stride()

second_point = points[1]
second_point.size()

second_point.stride()


points = torch.tensor([[400.0, 900.0], [50.0, 30.0], [2.0, 1.0]])
second_point = points[1]
second_point # second row
second_point[0] = 10.0
second_point
points

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1].clone()
second_point
second_point[0] = 10.0
points


## 3.8.2 Transposing without copying
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points

points_t = points.t()
points_t

id(points.storage()) == id(points_t.storage()) # same storage

points.stride()

points_t.stride()


## 3.8.3 Transposing in higher dimensions
some_t = torch.ones(3, 4, 5)
transpose_t = some_t.transpose(0, 2)
some_t.shape

transpose_t.shape

some_t.stride()

transpose_t.stride()


# 3.8.4 Contiguous tensors
points.is_contiguous()
points_t.is_contiguous()

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_t = points.t()
points_t

points_t.storage()

points_t.stride()

points_t_cont = points_t.contiguous()
points_t_cont

points_t_cont.stride()
points_t_cont.storage()