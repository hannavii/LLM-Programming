import torch
import time

device = "cuda"
x = torch.randn(10000, 10000)

# CPU → GPU
torch.cuda.synchronize()
start = time.time()

x_gpu = x.to(device)

torch.cuda.synchronize()
print("CPU → GPU:", time.time() - start)

# GPU computation
torch.cuda.synchronize()
start = time.time()

y_gpu = x_gpu * 2

torch.cuda.synchronize()
print("GPU computation:", time.time() - start)

# GPU → CPU
torch.cuda.synchronize()
start = time.time()

y_cpu = y_gpu.cpu()

torch.cuda.synchronize()
print("GPU → CPU:", time.time() - start)