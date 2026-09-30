import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)