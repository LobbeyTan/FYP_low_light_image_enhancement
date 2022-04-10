from numpy import outer
import torch
from torch.utils.data import DataLoader
from configs.option import Option
from data.custom_image_dataset import CustomImageDataset
from networks.discriminators import NLayerDiscriminator
from networks.generators import ResnetGenerator
from networks.networks import Activation, ConvLayer


if __name__ == "__main__":

    dataset = CustomImageDataset(
        img_dir="./datasets/summer2winter_yosemite",
        opt=Option(phase="train")
    )

    sample = torch.rand(10, 3, 224, 224)
    G = ResnetGenerator(3, 3)

    output = G(sample)

    print(output.shape)

    
