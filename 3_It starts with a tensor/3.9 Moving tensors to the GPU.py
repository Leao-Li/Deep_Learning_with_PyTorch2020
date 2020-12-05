# 3.9 Moving tensors to the GPU

import torch

points = torch.tensor([[20.0, 36.0], [45.0, 6.0], [26.0, 88.0]])

points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device='cuda')

points_gpu = points.to(device='cuda')

points_gpu = points.to(device='cuda:0')

points = 2 * points
points_gpu = 2 * points.to(device='cuda')

points_gpu = points_gpu + 4

points_cpu = points_gpu.to(device='cpu')

points_gpu = points.cuda()
points_gpu = points.cuda(0)
points_cpu = points_gpu.cpu()