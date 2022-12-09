from torch import nn
import torch


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        # The perceptual loss currently using:
        # VGG16 relu5_1

        # Conv_1: 2 layers
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv_2: 2 layers
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv_3: 3 layers
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv_4: 3 layers
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv_5: 3 layers
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, vgg_choose="relu5_1"):
        h = self.relu1_1(self.conv1_1(x))
        h = self.relu1_2(self.conv1_2(h))
        relu1_2 = h
        h = self.maxpool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        relu2_2 = h
        h = self.maxpool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        relu3_3 = h
        h = self.maxpool3(h)

        h = self.relu4_1(self.conv4_1(h))
        relu4_1 = h
        h = self.relu4_2(self.conv4_2(h))
        relu4_2 = h
        h = self.relu4_3(self.conv4_3(h))
        relu4_3 = h
        h = self.maxpool4(h)

        h = self.relu5_1(self.conv5_1(h))
        relu5_1 = h
        h = self.relu5_2(self.conv5_2(h))
        relu5_2 = h
        h = self.relu5_3(self.conv5_3(h))
        relu5_3 = h
        h = self.maxpool5(h)

        if vgg_choose == "relu1_2":
            return relu1_2
        if vgg_choose == "relu2_2":
            return relu2_2
        if vgg_choose == "relu3_3":
            return relu3_3
        if vgg_choose == "relu4_1":
            return relu4_1
        if vgg_choose == "relu4_2":
            return relu4_2
        if vgg_choose == "relu4_3":
            return relu4_3
        if vgg_choose == "relu5_1":
            return relu5_1
        if vgg_choose == "relu5_2":
            return relu5_2
        if vgg_choose == "relu5_3":
            return relu5_3

        return h


def vgg_preprocess(batch, vgg_mean=False):
    tensortype = type(batch.data)
    
    if batch.shape[1] == 1:
        batch = batch.repeat(1, 3, 1, 1)

    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    if vgg_mean:
        mean = tensortype(batch.data.size())
        mean[:, 0, :, :] = 103.939
        mean[:, 1, :, :] = 116.779
        mean[:, 2, :, :] = 123.680
        batch = batch.sub(torch.autograd.Variable(mean))  # subtract mean
    return batch


def load_vgg16(path, device=torch.device("cpu")):
    vgg = VGG16()
    vgg.to(device)
    vgg.load_state_dict(torch.load(path))

    return vgg
