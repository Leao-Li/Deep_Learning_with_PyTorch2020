# 3.4 Named tensors

import torch

img_t = torch.randn(3, 5, 5) # shape [channels, rows, columns]
img_t.shape
img_t.size()

weights = torch.tensor([0.2126, 0.7152, 0.0722])

batch_t = torch.randn(2, 3, 5, 5) # shape [batch, channels, rows, columns]
batch_t.size()
batch_t.shape

# the RGB channels are always in dimension -3
img_gray_naive = img_t.mean(-3)
img_gray_naive
img_gray_naive.shape

batch_gray_naive = batch_t.mean(-3)
batch_gray_naive
batch_gray_naive.shape

unsqueezed_weights = weights.unsqueeze(-1).unsqueeze_(-1)
unsqueezed_weights
img_weights = (img_t * unsqueezed_weights)
batch_weights = (batch_t * unsqueezed_weights)
img_gray_weighted = img_weights.sum(-3)
batch_gray_weighted = batch_weights.sum(-3)
batch_weights.shape, batch_t.shape, unsqueezed_weights.shape

img_gray_weighted_fancy = torch.einsum('...chw,c->...hw', img_t, weights)
batch_gray_weighted_fancy = torch.einsum('...chw,c->...hw', batch_t, weights)
batch_gray_weighted_fancy.shape

# named tensors 
# PyTorch 1.3 added named tensors as an experimental feature 
weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['channels'])
weights_named

img_named = img_t.refine_names(..., 'channels', 'rows', 'columns')
batch_named = batch_t.refine_names(..., 'channels', 'rows', 'columns')
print("img named:", img_named.shape, img_named.names)
print("batch named:", batch_named.shape, batch_named.names)

weights_aligned = weights_named.align_as(img_named)
weights_aligned.shape, weights_aligned.names

gray_named = (img_named * weights_aligned).sum('channels')
gray_named.shape, gray_named.names

gray_plain = gray_named.rename(None)
gray_plain.shape, gray_plain.names