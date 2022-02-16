import torch

if not torch.cuda.is_available():
    CUDA_DEVICE = "cpu"
elif torch.cuda.device_count() == 1:
    CUDA_DEVICE = "cuda:0"  # for Miku
else:
    CUDA_DEVICE = "cuda:3"  # for Yuhao

CROP_WIDTH, CROP_HEIGHT = 512, 512
