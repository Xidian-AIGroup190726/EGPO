import torch
from torch.nn.modules import Module, Conv2d, BatchNorm2d, ReLU
from torch.nn.functional import interpolate, adaptive_avg_pool2d


from model.backbone.ResNett import *


class FarNet(Module):
    def __init__(self, num_classes=7, num_feature=256, pretrained=False, ignore_index=255, **kwargs):
        super(FarNet, self).__init__()

        self.num_classes = num_classes
        self.num_feature = num_feature
        self.ignore_index = ignore_index
        self.EPS = 1e-5
        self.current_step = 0
        self.annealing_step = 2000
        self.focal_factor = 4
        self.focal_z = 1.0

        self.backbone = ResNet50()

        self.conv_c6 = Conv2d(2048, num_feature, 1)
        self.conv_c5 = Conv2d(2048, num_feature, 1)
        self.conv_c4 = Conv2d(1024, num_feature, 1)
        self.conv_c3 = Conv2d(512, num_feature, 1)
        self.conv_c2 = Conv2d(256, num_feature, 1)

        self.fs5 = FSModule(num_feature, num_feature)
        self.fs4 = FSModule(num_feature, num_feature)
        self.fs3 = FSModule(num_feature, num_feature)
        self.fs2 = FSModule(num_feature, num_feature)

        self.up5 = Decoder(num_feature, 8)
        self.up4 = Decoder(num_feature, 4)
        self.up3 = Decoder(num_feature, 2)
        self.up2 = Decoder(num_feature, 1)

        self.classify = Conv2d(num_feature, num_classes, 3, padding=1)

    def forward(self, x):
        c2, c3, c4, c5 = self.backbone(x)
        c6 = adaptive_avg_pool2d(c5, (1, 1))
        u = self.conv_c6(c6)

        p5 = self.conv_c5(c5)
        p4 = (self.conv_c4(c4) + interpolate(p5, scale_factor=2)) / 2.
        p3 = (self.conv_c3(c3) + interpolate(p4, scale_factor=2)) / 2.
        p2 = (self.conv_c2(c2) + interpolate(p3, scale_factor=2)) / 2.

        z5 = self.fs5(p5, u)
        z4 = self.fs4(p4, u)
        z3 = self.fs3(p3, u)
        z2 = self.fs2(p2, u)

        o5 = self.up5(z5)
        o4 = self.up4(z4)
        o3 = self.up3(z3)
        o2 = self.up2(z2)

        x = (o5 + o4 + o3 + o2) / 4.
        x = interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)
        logit = self.classify(x)
        return logit

    # def _get_loss(self, logit, label):
    #     logit = logit.permute(0, 2, 3, 1).flatten(0, 2)
    #     label = label.flatten()
    #     mask = label != self.ignore_index
    #     logit, label = logit[mask], label[mask]
    #     loss = torch.nn.functional.cross_entropy(logit, label, reduction="none")
    #
    #     probs = torch.softmax(logit, dim=1)
    #     index = torch.unsqueeze(label, 1)
    #     p = torch.gather(probs, 1, index).squeeze_()
    #
    #     z = torch.pow(1.0 - p, self.focal_factor)
    #     z = self.focal_z * z
    #
    #     if self.current_step < self.annealing_step:
    #         z = z + (1 - z) * (1 - self.current_step / self.annealing_step)
    #     self.current_step += 1
    #
    #     loss = z * loss
    #     avg_loss = torch.mean(loss) / (torch.mean(mask.type(torch.float32)) + self.EPS)
    #     return avg_loss




class Decoder(Module):
    def __init__(self, c_in, scale):
        super(Decoder, self).__init__()

        assert scale in [1, 2, 4, 8]

        if scale >= 1:
            self.conv1 = Conv2dBN(c_in, c_in, 3, padding=1)
        if scale >= 4:
            self.conv2 = Conv2dBN(c_in, c_in, 3, padding=1)
        if scale >= 8:
            self.conv3 = Conv2dBN(c_in, c_in, 3, padding=1)

        self.scale = scale

    def forward(self, x):
        if self.scale >= 1:
            x = self.conv1(x)
            if self.scale == 1:
                return x

        if self.scale >= 2:
            x = interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        if self.scale >= 4:
            x = self.conv2(x)
            x = interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        if self.scale >= 8:
            x = self.conv3(x)
            x = interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        return x


class FSModule(Module):
    def __init__(self, cv, cu):
        super(FSModule, self).__init__()

        self.conv1 = Conv2dBN(cv, cu, 1)
        self.conv2 = Conv2dBN(cv, cu, 1)

    def forward(self, v, u):
        x = self.conv1(v)
        r = torch.mul(x, u)
        k = self.conv2(v)
        z = k / (1 + torch.exp(-r))
        return z


class Conv2dBN(Module):
    def __init__(self, c_in, c_out, filter_size, stride=1, padding=0, **kwargs):
        super(Conv2dBN, self).__init__()
        self.conv = Conv2d(c_in, c_out, filter_size, stride=stride, padding=padding, **kwargs)
        self.bn = BatchNorm2d(c_out)
        self.relu = ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# if __name__ == '__main__':
#     net = FarNet()
#     x = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
#     lb = torch.zeros(1, 1, 224, 224, dtype=torch.int64)
#     y = net(x)
#     print(y.shape)
