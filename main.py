from pyexpat import model
from numpy import outer
import torch
from torch.utils.data import DataLoader
from configs.option import Option
from data.custom_image_dataset import CustomImageDataset
from models.cycleGAN import CycleGANModel
from networks.discriminators import NLayerDiscriminator
from networks.generators import ResnetGenerator
from networks.networks import Activation, ConvLayer, ResidualLayer


if __name__ == "__main__":

    dataset = CustomImageDataset(
        img_dir="./datasets/summer2winter_yosemite",
        opt=Option(phase="train")
    )

    batch_1 = torch.rand(10, 3, 224, 224)
    batch_2 = torch.rand(10, 3, 224, 224)

    model = CycleGANModel()

    model.save_model("./checkpoints", 1)
    
