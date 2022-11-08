import torch
import torch.nn as nn

loss_func = nn.CrossEntropyLoss()

a = torch.Tensor([1, 2, 3]).view(1, -1)
b = torch.nn.functional.softmax(a)
c = torch.nn.functional.softmax(b)
print(b, c)
