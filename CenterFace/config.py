import torch
from torchvision import transforms as T


class Config:
    # save pth
    checkpoints = "checkpoints"
    load_model = "500.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    threshold = 0.4

    # preprocess
    insize = [800, 800]
    channels = 3
    downscale = 4
    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]

    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])








