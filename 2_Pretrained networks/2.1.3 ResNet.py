from torchvision import models

dir(models)

resnet = models.resnet101(pretrained=True)
resnet


# we defined a preprocess function that will scale the 
# input image to 256x256, crop the image to 224x224 around center,
# transform it to a tensor,
# and normalize its RGB (red, green, blue) components so that they
# have defined means and standard deviations.
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
    ])


from PIL import Image
path = "E:/DeepLearningExperiment/Deep-Learning-with-PyTorch/dlwpt-code-master/data/p1ch2/bobby.jpg"
img = Image.open(path)
img
img.show()

img_t = preprocess(img)
img_t.size()

import torch
batch_t = torch.unsqueeze(img_t, 0)
batch_t.size()

resnet.eval() # 不启用 BatchNormalization 和 Dropout
# model.train()
# 启用 BatchNormalization 和 Dropout

out = resnet(batch_t)

with open('E:/DeepLearningExperiment/Deep-Learning-with-PyTorch/dlwpt-code-master/data/p1ch2/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

_, index = torch.max(out, 1)

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()

_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]