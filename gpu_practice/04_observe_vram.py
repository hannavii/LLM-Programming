import torch

print(
    torch.cuda.memory_allocated()
    / 1024**2, "MB"
)

x = torch.randn(
    10000,
    10000,
    device="cuda"
)

print(
    torch.cuda.memory_allocated()
    / 1024**2, "MB"
)