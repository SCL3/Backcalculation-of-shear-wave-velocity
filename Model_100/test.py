import torch
print(torch.__version__)
print(torch.amp.GradScaler("cuda", enabled=False))  # must not raise