from torch import nn
import torch
import torch.nn.functional as F
from models.vgg import vgg_preprocess


class GANLoss(nn.Module):

    def __init__(self, gan_mode="lsgan", target_real_label=1.0, target_fake_label=0.0) -> None:
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.gan_mode = gan_mode

        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def __call__(self, prediction, is_real=True) -> torch.Tensor:
        target_tensor = self.real_label if is_real else self.fake_label
        return self.loss(prediction, target_tensor.expand_as(prediction))


class SelfFeaturePreservingLoss(nn.Module):

    def __init__(self) -> None:
        super(SelfFeaturePreservingLoss, self).__init__()

        self.normalize = nn.InstanceNorm2d(512, affine=False)

    def __call__(self, vgg: nn.Module, img: torch.Tensor, target: torch.Tensor):
        img = vgg_preprocess(img)
        target = vgg_preprocess(target)

        loss = torch.mean((self.normalize(vgg(img)) -
                          self.normalize(vgg(target))) ** 2)

        return loss


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):

        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg, 2)
        Drb = torch.pow(mr-mb, 2)
        Dgb = torch.pow(mb-mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) +
                      torch.pow(Dgb, 2), 0.5)

        return k


class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor(
            [[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor(
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor(
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor(
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(
            data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff = torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor(
            [0.3]).cuda(), torch.FloatTensor([0]).cuda()), torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(
            enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool-org_pool)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E


class L_exp(nn.Module):

    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):

        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(
            torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]), 2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size


class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)

    def forward(self, x):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b, c, h, w = x.shape
        # x_de = x.cpu().detach().numpy()
        r, g, b = torch.split(x, 1, dim=1)
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) +
                      torch.pow(Dg, 2), 0.5)
        # print(k)

        k = torch.mean(k)
        return k
