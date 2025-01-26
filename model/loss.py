import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss


def dice_loss_f(inputs, targets):
    # 确保输入和目标形状相同
    assert inputs.shape[0] == targets.shape[0], "Batch size of inputs should be equal to targets"

    # 计算每个类别的 Dice Loss
    dice_loss = 0
    for i in range(inputs.shape[1]):
        iflat = inputs[:, i].view(inputs.size(0), -1)
        tflat = targets.view(targets.size(0), -1)

        # 计算Dice系数
        intersection = (iflat * tflat).sum(1)
        union = iflat.sum(1) + tflat.sum(1)

        # 避免除以零
        dice_score = (2. * intersection + 1e-6) / (union + 1e-6)
        dice_loss += 1 - dice_score.mean()

    return dice_loss

class LOSS_DATAIL(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LOSS_DATAIL, self).__init__()

        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)

        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[4. / 10], [3. / 10], [2. / 10], [1. / 10]],
                                                           dtype=torch.float32).reshape(1, 4, 1, 1).type(
            torch.cuda.FloatTensor))

    def forward(self, boundary_logits, gtmasks):

        # boundary_logits = boundary_logits.unsqueeze(1)
        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        #boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)

        boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=8, padding=1)
        boundary_targets_x8 = boundary_targets_x8.clamp(min=0)

        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')

        #boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0

        #boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0

        #boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0

        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up, boundary_targets_x8_up),
                                               dim=1)

        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)

        #boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0

        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)
        criterion = CrossEntropyLoss(ignore_index=255)
        boudary_targets_pyramid = boudary_targets_pyramid.squeeze(1).long()
        print(boudary_targets_pyramid)
        cri_loss = criterion(boundary_logits, boudary_targets_pyramid)
        dice_loss = 0#dice_loss_f(torch.sigmoid(boundary_logits), boudary_targets_pyramid)
        return cri_loss, dice_loss

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            nowd_params += list(module.parameters())
        return nowd_params

if __name__ =='__main__':
    input = torch.randn(2,3,512,512).cuda()
    mask = torch.randn(2,512,512).cuda()
    loss_datail = LOSS_DATAIL().cuda()
    out1 = loss_datail(input,mask)
    print(out1)