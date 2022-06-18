import torch
from torchvision import transforms as T


class Config:
    embedding_size = 512
    checkpoints = "checkpoints"
    load_detect_model = "490.pth"
    load_recog_model = "34.pth"
    member_path = "./Emroll_members"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    threshold = 0.35    # detection threshold
    recog_th = 0.4
    knn_num = 10
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

    input_shape = [3, 112, 112]
    recog_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    cap_index = 0




