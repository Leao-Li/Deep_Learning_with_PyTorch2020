# 3.10 NumPy interoperability

import torch
import numpy as np

# tensor to numpy
points = torch.ones(3, 4)
points_np = points.numpy()
points_np

# numpy to tensor
points = torch.from_numpy(points_np)