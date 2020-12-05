# 3.3 Indexing tensors

import torch

some_list = list(range(6))
print(some_list)

# index list
# All elements in the list
some_list[:] # index all 

# From element 1 inclusive to element 4 exclusive
some_list[1:4]

# From element 1 inclusvie to the end of the list
some_list[1:]

# From the start of the list to element 4 exclusive
some_list[:4]

# From the start of the list to one before the last element
some_list[: -1]

# From element 1 inclusive to element 4 exclusive, in steps 2
some_list[1: 4: 2] 

points = torch.tensor([[5.2, 9.5], [2.6, 4.0], [3.0, 2.5]])
print(points)

# All rows after the first; implicitly all columns
points[1:]

# All rows after the first; all columns
points[1:, :]

# All rows after the first; first column
points[1:, 0]

# Adds a dimension of size 1, just like unsequeeze
points[None]
points[None].size()