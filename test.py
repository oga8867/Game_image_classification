import torch
a = torch.randn(1,6)
print(a)

b = torch.max(a,1)
print(b)