import torch
import torch.nn as nn
import torchvision
import random
import torch.nn.functional as Fun

class Discriminator_VGG_128_bn(nn.Module):
    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_128_bn, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [256, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [512, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)

        self.linear1 = nn.Linear(512 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))

        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_128, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = nn.Conv2d(in_nc*3, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        # [256, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        # [512, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)

        self.linear1 = nn.Linear(nf * 8, 100)
        self.linear2 = nn.Linear(100, 1)
        self.pool=nn.AdaptiveAvgPool2d(1)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.conv0_1(fea))

        fea = self.lrelu(self.conv1_0(fea))
        fea = self.lrelu(self.conv1_1(fea))

        fea = self.lrelu(self.conv2_0(fea))
        fea = self.lrelu(self.conv2_1(fea))

        fea = self.lrelu(self.conv3_0(fea))
        fea = self.lrelu(self.conv3_1(fea))

        fea = self.lrelu(self.conv4_0(fea))
        fea = self.lrelu(self.conv4_1(fea))

        fea=self.pool(fea)
        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


class Discriminator_VGG_128_1(nn.Module):
    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_128_1, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = nn.Conv2d(in_nc*3, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        # [256, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        # [512, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)

        self.linear1 = nn.Linear(nf * 8, 128)
        self.linear2 = nn.Linear(128, 1)
        self.pool=nn.AdaptiveAvgPool2d(1)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        ########################
        self.conv0_01 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_11 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        # [64, 64, 64]
        self.conv1_01 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.conv1_11 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        # [128, 32, 32]
        self.conv2_01 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.conv2_11 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        # [256, 16, 16]
        self.conv3_01 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.conv3_11 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        # [512, 8, 8]
        self.conv4_01 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.conv4_11 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)

        self.linear11 = nn.Linear(nf * 8, 128)
        self.linear21 = nn.Linear(128, 1)

    def forward(self, x1, x2, x3):
        x=torch.cat([x1, x2, x3], dim=1)
        ## x=Fun.interpolate(x, size=(128, 128),mode='bilinear')
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.conv0_1(fea))

        fea = self.lrelu(self.conv1_0(fea))
        fea = self.lrelu(self.conv1_1(fea))

        fea = self.lrelu(self.conv2_0(fea))
        fea = self.lrelu(self.conv2_1(fea))

        fea = self.lrelu(self.conv3_0(fea))
        fea = self.lrelu(self.conv3_1(fea))

        fea = self.lrelu(self.conv4_0(fea))
        fea = self.lrelu(self.conv4_1(fea))

        fea=self.pool(fea)
        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)

        ######
        height=x1.shape[2]
        width=x1.shape[3]
        h_start=random.randint(0, height-128)
        w_start=random.randint(0, width-128)
        x_input=x1[:, :, h_start:h_start+128, w_start:w_start+128]
        ### x_input=Fun.interpolate(x1, size=(128, 128),mode='bilinear')
        fea1 = self.lrelu(self.conv0_01(x_input))
        fea1 = self.lrelu(self.conv0_11(fea1))

        fea1 = self.lrelu(self.conv1_01(fea1))
        fea1 = self.lrelu(self.conv1_11(fea1))

        fea1 = self.lrelu(self.conv2_01(fea1))
        fea1 = self.lrelu(self.conv2_11(fea1))

        fea1 = self.lrelu(self.conv3_01(fea1))
        fea1 = self.lrelu(self.conv3_11(fea1))

        fea1 = self.lrelu(self.conv4_01(fea1))
        fea1 = self.lrelu(self.conv4_11(fea1))

        fea1=self.pool(fea1)
        fea1 = fea1.view(fea1.size(0), -1)
        fea1 = self.lrelu(self.linear11(fea1))
        out1 = self.linear21(fea1)
        return out, out1


class Discriminator_VGG_128_2(nn.Module):
    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_128_2, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        # [256, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.conv0_1(fea))

        fea = self.lrelu(self.conv1_0(fea))
        fea = self.lrelu(self.conv1_1(fea))

        fea = self.lrelu(self.conv2_0(fea))
        fea = self.lrelu(self.conv2_1(fea))

        fea = self.lrelu(self.conv3_0(fea))
        out = self.lrelu(self.conv3_1(fea))

        return out

    def forward_feature(self, x):
        feature_list = []
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.conv0_1(fea))
        feature_list.append(fea)

        fea = self.lrelu(self.conv1_0(fea))
        fea = self.lrelu(self.conv1_1(fea))
        feature_list.append(fea)

        fea = self.lrelu(self.conv2_0(fea))
        fea = self.lrelu(self.conv2_1(fea))
        feature_list.append(fea)
        return feature_list

'''
from models.archs.transformer.Models import Discriminator_transformer

class Discriminator_trans(nn.Module):
    def __init__(self, ):
        super(Discriminator_trans, self).__init__()
        self.transformer = Discriminator_transformer(d_model=512, d_inner=1024, n_layers=4)

    def forward(self, x, mask):
        fea = self.transformer.forward(x, mask)
        return fea
'''

class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output

