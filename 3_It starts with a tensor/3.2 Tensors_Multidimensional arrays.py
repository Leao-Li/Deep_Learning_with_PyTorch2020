# 3.2.1 From Python lists to PyTorch tensors

a = [1.0, 2.0, 1.0]

a[0]

# Change the value of list a
a[2] = 3.0

a


# 3.2.2 Constructing our first tensors

# Import the torch module
import torch

# Create a one-dimensional tensor of size 3 filled with1s
a = torch.ones(3)
print(a)
a.size() # the size of tensor a

a[1]

float(a[1])

a[2] = 2.0
print(a)


# 3.2.3 The essence of tensors
points = torch.zeros(6)
points[0] = 4.0
points[1] = 1.0
points[2] = 5.0
points[3] = 3.0
points[4] = 2.0
points[5] = 1.0

print(points)

# we can also pass a Python list to the constructor, to the same effect:
points = torch.tensor([4.0, 1.0, 5.0, 3.0, 2.0, 1.0])
print(points)

# To get the coordinates of the first point, we do the following:
float(points[0]), float(points[1])

# Creat a two-dimensional tensor
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
print(points)
points.shape
points.size()

# Create 2D tensor with 0
points = torch.zeros(3, 2)
points

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
print(points)

# index
points[0, 1] # 0 row, 1 column
points[0] # 0 row












