import torch

x32 = torch.randn(
    5000,
    5000,
    device="cuda",
    dtype=torch.float32
)

memory32 = (
    x32.element_size()
    * x32.nelement()
)

print(memory32 / 1024**2, "MB")

x16 = torch.randn(
    5000,
    5000,
    device="cuda",
    dtype=torch.float16
)

memory16 = (
    x16.element_size()
    * x16.nelement()
)

print(memory16 / 1024**2, "MB")