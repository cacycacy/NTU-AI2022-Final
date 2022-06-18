import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from Config import Config as conf

class Load_Enroll_Dataset(Dataset):
    def __init__(self, images_dir):
        self.total_imgs, self.items = self.load_identities(images_dir)

    def __len__(self):
            return self.total_imgs

    def __getitem__(self, index):
        img_path = self.items[index]
        image = cv2.imread(img_path)
        if image.shape[0] != 112:
            image = cv2.resize(image, (112, 112))
        image = conf.recog_transforms(image)
        return img_path, image

    def load_identities(self, images_dir):
        ids = os.listdir(images_dir)
        items = []
        total_imgs = 0
        for label, id in enumerate(ids):
            imgs_clean = os.listdir(f"{images_dir}/{id}")
            total_imgs += len(imgs_clean)
            for img_clean in imgs_clean:
                # img_name, id
                item = f"{images_dir}/{id}/{img_clean}"
                items.append(item)

        return total_imgs, items

if __name__ == "__main__":
    data_path = "../../Member/"

    datasets = Load_Enroll_Dataset(data_path)
    loader = DataLoader(dataset=datasets, batch_size=1, shuffle=False)
    for img_path, img in loader:
        # print(data[0].shape)
        print(img_path)

        # break