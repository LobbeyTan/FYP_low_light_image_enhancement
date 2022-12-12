import os
from random import randint
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from configs.option import Option


class CustomImageDataset(Dataset):

    def __init__(self, img_dir, opt: Option) -> None:
        super(CustomImageDataset, self).__init__()

        self.opt = opt
        self.dir_A = os.path.join(img_dir, opt.phase + 'A' if opt.dir_A is None else opt.dir_A)
        self.dir_B = os.path.join(img_dir, opt.phase + 'B' if opt.dir_B is None else opt.dir_B)

        self.imgs_A, self.paths_A = self.extractImages(self.dir_A)
        self.imgs_B, self.paths_B = self.extractImages(self.dir_B)

        self.size_A = len(self.imgs_A)
        self.size_B = len(self.imgs_B)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    size=286, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(size=256) if self.opt.phase == "train" else transforms.CenterCrop(size=256),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return max(self.size_A, self.size_B)

    def __getitem__(self, index: int):

        idx_A = index % self.size_A
        idx_B = randint(0, self.size_B - 1)

        img_A = self.imgs_A[idx_A]
        img_B = self.imgs_B[idx_B]

        path_A = self.paths_A[idx_A]
        path_B = self.paths_B[idx_B]

        transformed_img_A = self.transform(img_A)
        transformed_img_B = self.transform(img_B)

        gray_A = self.getGrayImage(transformed_img_A)

        return {
            'img_A': self.getGray3DImage(transformed_img_A) if self.opt.grayscale else transformed_img_A,
            'img_A*': transformed_img_A,
            'img_B': transformed_img_B,
            'gray_A': gray_A,
            'path_A': path_A,
            'path_B': path_B,
        }

    def extractImages(self, dir):
        paths = []
        images = []

        for (root, _, files) in os.walk(dir):
            for file in files:
                path = os.path.join(root, file)
                img = Image.open(path).convert('RGB')

                paths.append(path)
                images.append(img)

        return images, paths

    def getGrayImage(self, image: torch.Tensor):
        r, g, b = image[0] + 1, image[1] + 1, image[2] + 1
        grayscale = 1. - (0.299*r + 0.587*g + 0.114*b) / 2.
        return torch.unsqueeze(grayscale, dim=0)

    def getGray3DImage(self, image: torch.Tensor):
        r, g, b = image[0], image[1], image[2]
        grayscale = 0.299*r + 0.587*g + 0.114*b

        output = torch.unsqueeze(grayscale, dim=0)

        # output = output.repeat(3, 1, 1)
            
        return output
