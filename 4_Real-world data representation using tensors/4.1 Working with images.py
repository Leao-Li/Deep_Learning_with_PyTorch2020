# 4.1 Working with images

# 4.1.2 Loading an image file
import torch
from PIL import Image
import numpy as np

path = 'E:/DeepLearningExperiment/Deep-Learning-with-PyTorch/dlwpt-code-master/data/p1ch4/image-dog/bobby.jpg'
img = Image.open(path) # read an image from a file

img_arr = np.array(img)
img_arr.shape

# additionally, you can also directly use imageio module
import imageio
img_arr = imageio.imread(path)
img_arr.shape

# image numpy array to PyTorch tensor
img_t = torch.from_numpy(img_arr)
img_t.size()

# Changing the layout, HxWxC to CxHxW
out = img_t.permute(2, 0, 1)
out.size()

batch_size = 3
batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)


import os

data_dir = 'E:/DeepLearningExperiment/Deep-Learning-with-PyTorch/dlwpt-code-master/data/p1ch4/image-cats/'
filenames = [name for name in os.listdir(data_dir)
             if os.path.splitext(name)[-1] == '.png']
for i, filename in enumerate(filenames):
    img_arr = imageio.imread(os.path.join(data_dir, filename))
    img_t = torch.from_numpy(img_arr)
    img_t = img_t.permute(2, 0, 1) # (0, 1, 2) -> (3, 0, 1)
    img_t = img_t[:3]
    batch[i] = img_t


# Normalizing the data
# the input data ranges roughly from 0 to 1, or from -1 to 1 
batch = batch.float()
batch /= 255.0

n_channels = batch.shape[1] # output: 3
n_channels

for c in range(n_channels):
    mean = torch.mean(batch[:, c])
    std = torch.std(batch[:, c])
    batch[:, c] = (batch[:, c] - mean) / std
    print(batch[:, c].shape)