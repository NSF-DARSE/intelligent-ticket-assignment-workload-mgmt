import torch
from torch.profiler import profile, ProfilerActivity

x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)

with profile(activities=[ProfilerActivity.CPU]) as prof:
    z = torch.matmul(x, y)

print(prof.key_averages().table(sort_by="cpu_time_total"))