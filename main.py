from pyexpat import model
from numpy import outer
import torch
from torch.utils.data import DataLoader
from configs.option import Option
from data.custom_image_dataset import CustomImageDataset
from models.cycleGAN import CycleGANModel
from models.enlighten import EnlightenGAN
from models.vgg import VGG16
from networks.discriminators import NLayerDiscriminator
from networks.generators import ResnetGenerator, UnetGenerator
from networks.networks import Activation, ConvLayer, ResidualLayer


if __name__ == "__main__":



    sample_data = torch.rand(size=(1, 3, 256, 256))
    
    for data in sample_data:
        r,g,b = data[0]+1, data[1]+1, data[2]+1
        A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
        A_gray = torch.unsqueeze(A_gray, 0).reshape(1, 1, 256, 256)
    
    G = EnlightenGAN()
