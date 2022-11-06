import torch

a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([1, 2, 1, 2])

a += b

print(a)
