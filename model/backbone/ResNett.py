import torchvision
import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F

__all__ = ["ResNet18", "ResNet50", "ResNet34", "ResNet101"]

class ResNet18(nn.Module):
    output_size = 512

    def __init__(self, pretrained=True):
        super(ResNet18, self).__init__()
        pretrained = torchvision.models.resnet18(pretrained=pretrained)
        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)

        return b1, b2, b3, b4



class ResNet34(nn.Module):
    output_size = 512

    def __init__(self, pretrained=True):
        super(ResNet34, self).__init__()
        pretrained = torchvision.models.resnet34(pretrained=pretrained)

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)

        return b1, b2, b3, b4



class ResNet50(nn.Module):
    output_size = 2048

    def __init__(self):
        super(ResNet50, self).__init__()

        local_weights_path = './pretrained/resnet50.pth'
        # 加载预训练模型
        self.resnet50_model = models.resnet50()
        # 加载本地权重
        state_dict = torch.load(local_weights_path)
        self.resnet50_model.load_state_dict(state_dict, strict=False)

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]:
            self.add_module(module_name, getattr(self.resnet50_model, module_name))

    def forward(self, x):
        b0 = self.relu(self.bn1(self.conv1(x)))
        b = self.maxpool(b0)
        b1 = self.layer1(b)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)

        return b1, b2, b3, b4



class ResNet101(nn.Module):
    output_size = 2048

    def __init__(self):
        super(ResNet101, self).__init__()

        local_weights_path = './pretrained/resnet101.pth'
        # 加载预训练模型
        self.resnet101_model = models.resnet101()
        # 加载本地权重
        state_dict = torch.load(local_weights_path)
        self.resnet101_model.load_state_dict(state_dict, strict=False)


        # 将 resnet101_model 中的层添加到当前模型中
        for module_name in [
            "conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"
        ]:
            self.add_module(module_name, getattr(self.resnet101_model, module_name))

    def forward(self, x):
        # 注意：这里需要确保 forward 方法使用 self 来调用层
        b0 = self.relu(self.bn1(self.conv1(x)))
        b = self.maxpool(b0)
        b1 = self.layer1(b)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)

        return b1, b2, b3, b4








class resnext50_32x4d(nn.Module):
    output_size = 2048

    def __init__(self, pretrained=True):
        super(resnext50_32x4d, self).__init__()
        pretrained = torchvision.models.resnext50_32x4d(pretrained=pretrained)

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x, get_ha=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)

        if get_ha:
            return b1, b2, b3, b4, pool

        return pool


class resnet152(nn.Module):
    output_size = 2048

    def __init__(self, pretrained=True):
        super(resnet152, self).__init__()
        pretrained = torchvision.models.resnet152(pretrained=pretrained)

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x, get_ha=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)

        if get_ha:
            return b1, b2, b3, b4, pool

        return pool

if __name__ == "__main__":
    from thop import profile
    x = torch.autograd.Variable(torch.randn(1, 3, 512, 512))
    net = ResNet50()
    print(net)
    out = net(x)
    print(out[0].shape,out[1].shape,out[2].shape,out[3].shape)
    flops, params = profile(net, (x,))
    print('flops: ', flops, 'params: ', params)
