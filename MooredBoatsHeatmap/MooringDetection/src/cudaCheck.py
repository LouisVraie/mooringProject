import torch
import os

print(f'PyTorch version: {torch.__version__}')
print('*'*10)
print('_CUDA version: ')
print(os.system("nvcc --version"))
print('*'*10)
print(f'CUDNN version: {torch.backends.cudnn.version()}')
print(f'Available GPU devices: {torch.cuda.device_count()}')
print(f'Device Name: {torch.cuda.get_device_name()}')

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
