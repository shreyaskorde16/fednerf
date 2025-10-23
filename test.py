import torch

# Check if PyTorch can even see the GPU
print(f"CUDA Available: {torch.cuda.is_available()}")

# CUDA version PyTorch was built with (e.g., '12.1')
print(f"PyTorch CUDA Version: {torch.version.cuda}")

# cuDNN version PyTorch is using (e.g., '8902')
if torch.backends.cudnn.is_available():
    print(f"PyTorch cuDNN Enabled: {torch.backends.cudnn.enabled}")
    print(f"PyTorch cuDNN Version: {torch.backends.cudnn.version()}")
else:
    print("cuDNN not available/enabled.")