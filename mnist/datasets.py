import os
import cv2
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class numberDataset(Dataset):
    def __init__(self, path, train=True, transform=None):
        assert os.path.isdir(path), path + " is not directory"

        self.x_data = []
        self.y_data = []

        for label in os.listdir(path):
            for filename in os.listdir(os.path.join(path, label)):
                img = cv2.imread(
                    os.path.join(path, label, filename), cv2.IMREAD_GRAYSCALE
                )

                # img_binary = cv2.adaptiveThreshold(
                #     img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
                # )
                # kernel = np.ones((3,3), np.uint8)
                # img_binary = cv2.erode(img_binary, kernel, iterations=1)
                # img_binary = cv2.dilate(img_binary, kernel, iterations=2)

                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                # img = self.randomPadding(img, 5)
                if train:
                    img = self.randomRotate(img, 45)
                # img = self.randomCrop(img)
                img = img / 255
                img = self.resize(img, (32, 32))

                if transform:
                    img = transform(img)

                self.x_data.append(img)
                self.y_data.append(int(label))

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def randomPadding(self, img, padding):
        pad = random.randint(0, padding)

        newimg = (
            np.ones((img.shape[0] + pad * 2, img.shape[1] + pad * 2, img.shape[2]))
            * 255
        )
        newimg[pad : img.shape[0] + pad, pad : pad + img.shape[1], :] = img

        return newimg

    # def Padding(self, img, padding=8):
    #     pad = np.random.randint(0, padding+1, 4)
    #     new_shape = [img.shape[0] + pad[0] + pad[2], img.shape[1] + pad[1] + pad[3], img.shape[2]]
    #     newimg = np.ones((new_shape)) * 255
    #     newimg[pad[0]:img.shape[0] + pad[0], pad[1]:img.shape[1] + pad[1]] = img

    #     return img
    def resize(self, img, size):
        assert len(size) == 2, "variable size form should be (widht, height)"

        padding = abs(img.shape[0] - img.shape[1]) // 2
        if img.shape[0] > img.shape[1]:
            newimg = np.ones((img.shape[0], img.shape[0], img.shape[2]))
            newimg[:, padding : padding + img.shape[1], :] = img
        else:
            newimg = np.ones((img.shape[1], img.shape[1], img.shape[2]))
            newimg[padding : padding + img.shape[0], :, :] = img

        return cv2.resize(newimg, size)

    def randomRotate(self, img, angle=0):
        rotate = random.randint(-angle, angle)
        newimg = Image.fromarray(np.array(img, dtype="uint8"))
        newimg = newimg.rotate(rotate, expand=1, fillcolor=((255, 255, 255)))
        newimg = np.asarray(newimg, dtype=float)

        return newimg

    def randomCrop(self, img):
        select_crop = random.randint(0, 4)
        width_edge, height_edge = int(img.shape[0] * 0.1), int(img.shape[1] * 0.1)
        if select_crop == 0:
            return img[0 : img.shape[1] - height_edge, 0 : img.shape[0] - width_edge, :]
        elif select_crop == 1:
            return img[height_edge:, 0 : img.shape[0] - width_edge, :]
        elif select_crop == 2:
            return img[0 : img.shape[1] - height_edge :, width_edge:, :]
        elif select_crop == 3:
            return img[height_edge:, width_edge:, :]
        elif select_crop == 4:
            return img[
                height_edge // 2 : img.shape[1] + height_edge // 2,
                width_edge // 2 : img.shape[0] + width_edge // 2,
                :,
            ]


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    dataset = numberDataset("datasets/numbers/train", transform=transform)
    dataLoader = DataLoader(dataset, batch_size=2, shuffle=True)

    for data in dataLoader:
        print(data)
