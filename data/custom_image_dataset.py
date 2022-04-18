import os
from random import randint
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from configs.option import Option


class CustomImageDataset(Dataset):

    def __init__(self, img_dir, opt: Option) -> None:
        super(CustomImageDataset, self).__init__()

        self.dir_A = os.path.join(img_dir, opt.phase + 'A')
        self.dir_B = os.path.join(img_dir, opt.phase + 'B')

        self.imgs_A, self.paths_A = self.extractImages(self.dir_A)
        self.imgs_B, self.paths_B = self.extractImages(self.dir_B)

        self.size_A = len(self.imgs_A)
        self.size_B = len(self.imgs_B)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    size=286, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(size=256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ]
        )

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

    def __len__(self):
        return max(self.size_A, self.size_B)

    def __getitem__(self, index: int):

        idx_A = index % self.size_A
        idx_B = randint(0, self.size_B - 1)

        img_A = self.imgs_A[idx_A]
        img_B = self.imgs_B[idx_B]

        path_A = self.paths_A[idx_A]
        path_B = self.paths_B[idx_B]

        return {
            'img_A': self.transform(img_A),
            'img_B': self.transform(img_B),
            'path_A': path_A,
            'path_B': path_B,
        }
