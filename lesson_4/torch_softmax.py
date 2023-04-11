import torch
from torch import nn
assert (torch.has_cuda and torch.cuda.is_available()) is True

d = torch.device('cuda:0' if torch.has_cuda else 'cpu')
assert torch.cuda.current_device() == 0
m, n = 1335, 16

s = nn.Softmax(dim=-1)
i = torch.randn(m, n, dtype=torch.float, device=d, requires_grad=False)
o = torch.zeros(m, n, dtype=torch.float, device=d, requires_grad=False)

for _ in range(300):
    o = s(i)

print(o)
