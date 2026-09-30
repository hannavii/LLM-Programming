import torch
import time

N = 50000

A = torch.randn(N, N)
B = torch.randn(N, N)

start = time.time()
C = torch.matmul(A, B)
print(
    "CPU time:",
    time.time() - start
)

A = A.cuda()
B = B.cuda()

torch.cuda.synchronize()
start = time.time()

C = torch.matmul(A, B)

torch.cuda.synchronize()
print(
    "GPU time:",
    time.time() - start
)