# 3.5 Tensor element types

import torch

double_points = torch.ones(10, 2, dtype=torch.double)
print(double_points)
double_points.dtype


short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)
print(short_points)
short_points.dtype

double_points = torch.zeros(10, 2).double()
short_points = torch.ones(10, 2).short()

double_points = torch.zeros(10, 2).to(torch.double)
short_points = torch.ones(10, 2).to(dtype=torch.short)

points_64 = torch.rand(5, dtype=torch.double)
points_short = points_64.to(torch.short)
points_64 * points_short # works from PyTorch 1.3 onwards