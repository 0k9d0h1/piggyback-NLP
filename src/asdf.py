import torch

a = torch.rand(2, 768, 512)

print(a[:, None, :, :].shape)
