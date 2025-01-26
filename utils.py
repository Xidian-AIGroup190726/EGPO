import math
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_cutmix_mask(img_size, ratio=2):
    cut_area = img_size[0] * img_size[1] / ratio
    w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
    h = np.round(cut_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)
    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0

    return mask.long()

def generate_unsup_aug_sc(conf_w, mask_w, data_s):
    b, _, im_h, im_w = data_s.shape
    device = data_s.device
    new_conf_w, new_mask_w, new_data_s = [], [], []
    for i in range(b):
        augmix_mask = generate_cutmix_mask([im_h, im_w]).to(device)
        new_conf_w.append((conf_w[i] * augmix_mask + conf_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_mask_w.append((mask_w[i] * augmix_mask + mask_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_data_s.append((data_s[i] * augmix_mask + data_s[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
    new_conf_w, new_mask_w, new_data_s = (torch.cat(new_conf_w), torch.cat(new_mask_w), torch.cat(new_data_s))

    return new_conf_w, new_mask_w, new_data_s

def generate_unsup_aug_ds(data_s1, data_s2):
    b, _, im_h, im_w = data_s1.shape
    device = data_s1.device
    new_data_s = []
    for i in range(b):
        augmix_mask = generate_cutmix_mask([im_h, im_w]).to(device)
        new_data_s.append((data_s1[i] * augmix_mask + data_s2[i] * (1 - augmix_mask)).unsqueeze(0))
    new_data_s = torch.cat(new_data_s)

    return new_data_s

def generate_unsup_aug_dc(conf_w, mask_w, data_s1, data_s2):
    b, _, im_h, im_w = data_s1.shape
    device = data_s1.device
    new_conf_w, new_mask_w, new_data_s = [], [], []
    for i in range(b):
        augmix_mask = generate_cutmix_mask([im_h, im_w]).to(device)
        new_conf_w.append((conf_w[i] * augmix_mask + conf_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_mask_w.append((mask_w[i] * augmix_mask + mask_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_data_s.append((data_s1[i] * augmix_mask + data_s2[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
    new_conf_w, new_mask_w, new_data_s = (torch.cat(new_conf_w), torch.cat(new_mask_w), torch.cat(new_data_s))

    return new_conf_w, new_mask_w, new_data_s

def generate_unsup_aug_sdc(conf_w, mask_w, data_s1, data_s2):
    b, _, im_h, im_w = data_s1.shape
    device = data_s1.device
    new_conf_w, new_mask_w, new_data_s = [], [], []
    for i in range(b):
        augmix_mask = generate_cutmix_mask([im_h, im_w]).to(device)
#        new_conf_w.append((conf_w[i] * augmix_mask + conf_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_mask_w.append((mask_w[i] * augmix_mask + mask_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        if i % 2 == 0:
            new_data_s.append((data_s1[i] * augmix_mask + data_s2[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        else:
            new_data_s.append((data_s2[i] * augmix_mask + data_s1[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))

    new_mask_w, new_data_s = torch.cat(new_mask_w), torch.cat(new_data_s)

    return new_conf_w, new_mask_w, new_data_s



def sdc_new(conf_w, mask_w, data_s1, data_s2):
    b, _, im_h, im_w = data_s1.shape
    device = data_s1.device
    new_conf_w, new_mask_w, new_data_s = [], [], []
    for i in range(b):
        augmix_mask = generate_cutmix_mask([im_h, im_w]).to(device)
        if i % 2 == 0:
            new_data_s.append((data_s1[i] * augmix_mask + data_s2[i] * (1 - augmix_mask)).unsqueeze(0))
        else:
            new_data_s.append((data_s2[i] * augmix_mask + data_s1[i] * (1 - augmix_mask)).unsqueeze(0))

    new_data_s = torch.cat(new_data_s)

    return conf_w, mask_w, new_data_s


def entropy_map(a, dim):
    em = - torch.sum(a * torch.log2(a + 1e-10), dim=dim)
    return em

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return iu, np.nanmean(iu)


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

    elif dataset == 'GID-15':
        cmap[0] = np.array([200, 0, 0])
        cmap[1] = np.array([250, 0, 150])
        cmap[2] = np.array([200, 150, 150])
        cmap[3] = np.array([250, 150, 150])
        cmap[4] = np.array([0, 200, 0])
        cmap[5] = np.array([150, 250, 0])
        cmap[6] = np.array([150, 200, 150])
        cmap[7] = np.array([200, 0, 200])
        cmap[8] = np.array([150, 0, 250])
        cmap[9] = np.array([150, 150, 250])
        cmap[10] = np.array([250, 200, 0])
        cmap[11] = np.array([200, 200, 0])
        cmap[12] = np.array([0, 0, 200])
        cmap[13] = np.array([0, 150, 200])
        cmap[14] = np.array([0, 200, 250])

    elif dataset == 'iSAID':
        cmap[0] = np.array([0, 0, 63])
        cmap[1] = np.array([0, 63, 63])
        cmap[2] = np.array([0, 63, 0])
        cmap[3] = np.array([0, 63, 127])
        cmap[4] = np.array([0, 63, 191])
        cmap[5] = np.array([0, 63, 255])
        cmap[6] = np.array([0, 127, 63])
        cmap[7] = np.array([0, 127, 127])
        cmap[8] = np.array([0, 0, 127])
        cmap[9] = np.array([0, 0, 191])
        cmap[10] = np.array([0, 0, 255])
        cmap[11] = np.array([0, 191, 127])
        cmap[12] = np.array([0, 127, 191])
        cmap[13] = np.array([0, 127, 255])
        cmap[14] = np.array([0, 100, 155])

    elif dataset == 'MSL' or dataset == 'MER':
        cmap[0] = np.array([128, 0, 0])
        cmap[1] = np.array([0, 128, 0])
        cmap[2] = np.array([128, 128, 0])
        cmap[3] = np.array([0, 0, 128])
        cmap[4] = np.array([128, 0, 128])
        cmap[5] = np.array([0, 128, 128])
        cmap[6] = np.array([128, 128, 128])
        cmap[7] = np.array([64, 0, 0])
        cmap[8] = np.array([192, 0, 0])

    elif dataset == 'Vaihingen' or dataset == 'Potsdam':
        cmap[0] = np.array([255, 255, 255])
        cmap[1] = np.array([0, 0, 255])
        cmap[2] = np.array([0, 255, 255])
        cmap[3] = np.array([0, 255, 0])
        cmap[4] = np.array([255, 255, 0])

    elif dataset == 'MF_DFC22':
        cmap[0] = np.array([219, 95, 87])
        cmap[1] = np.array([219, 151, 87])
        cmap[2] = np.array([219, 208, 87])
        cmap[3] = np.array([173, 219, 87])
        cmap[4] = np.array([117, 219, 87])
        cmap[5] = np.array([123, 196, 123])
        cmap[6] = np.array([88, 177, 88])
        cmap[7] = np.array([0, 128, 0])
        cmap[8] = np.array([88, 176, 167])
        cmap[9] = np.array([153, 93, 19])
        cmap[10] = np.array([87, 155, 219])
        cmap[11] = np.array([0, 98, 255])


    return cmap


def cos_matrix(a, b):

    batch,n = a.size(0),a.size(1)
    # 确保输入张量是浮点数类型
    a = a.type(torch.float32)
    b = b.type(torch.float32)

    # 对a和b进行归一化
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)

    x_length = (torch.sqrt(torch.sum(a_norm * a_norm, dim=1)).unsqueeze(1)).expand(-1, n, -1, -1)
    y_length = (torch.sqrt(torch.sum(b_norm * b_norm, dim=1)).unsqueeze(1)).expand(-1, n, -1, -1)

    a_norm = a_norm / x_length
    b_norm = b_norm / y_length

    cos_ = torch.sum(a_norm * b_norm, dim=1)

    chunks = torch.chunk(cos_, chunks=batch, dim=0)

    normalized_chunks = []

    for chunk in chunks:

        recombined_tensor = chunk.unsqueeze(0) # 变为 (1, 1, H, W)

        # 定义一个 3x3 的平均卷积核cccc
        kernel = torch.ones((1, 1, 3, 3)) / 9.0

        # 使用卷积操作
        average_recombined_tensor = F.conv2d(recombined_tensor, kernel, padding=1)

        # 去掉多余的维度
        average_recombined_tensor = average_recombined_tensor.squeeze(0)

        # 计算最小值和最大值
        min_val = average_recombined_tensor.min()
        max_val = average_recombined_tensor.max()

        # 应用最小-最大归一化
        normalized_chunk = (average_recombined_tensor - min_val) / (max_val - min_val)

        # 将归一化后的张量添加到列表中
        normalized_chunks.append(normalized_chunk)

    # 现在使用torch.cat()来拼接所有归一化后的张量
    recombined_tensor = torch.cat(normalized_chunks, dim=0)

    return recombined_tensor

# x = torch.randn(4,15, 224, 224)
# y = torch.randn(4,15, 224, 224)
# # 应用cutmix
# data = cos_matrix(x, y)
# print(data.shape)


def cosine_similarity(a, b):

    a = a.view(-1).float()
    b = b.view(-1).float()

    # 计算两个张量的点积
    dot_product = torch.sum(a * b)

    # 计算两个张量的模长
    norm_a = torch.sqrt(torch.sum(a ** 2))
    norm_b = torch.sqrt(torch.sum(b ** 2))

    # 计算余弦相似度
    cosine_similarity = dot_product / (norm_a * norm_b)

    return cosine_similarity

def cutmix(data, data_0, mask1, mask2, pred1, pred2, alpha=0.2, p=0.5, size = 1):


    #alpha = (cosine_similarity(mask1, mask2) + 1) / 2
    #alpha = torch.mean(cos_matrix(pred1, pred2)).item()


    b = data.shape[0]
    # 图像尺寸
    h, w = data.size()[-2:]
    num_x = h // size
    num_y = w // size
    e = []
    for k in range(b):


        # 遍历图像的每个块
        for i in range(size):
            for j in range(size):
                # 计算块的坐标
                start_x = j * num_x
                start_y = i * num_y
                end_x = start_x + num_x
                end_y = start_y + num_y

                # 从原图中裁剪出小块
                data_1 = pred1[k, :, start_x:end_x, start_y:end_y]
                data_2 = pred2[k, :, start_x:end_x, start_y:end_y]


                alpha = torch.mean(cos_matrix(data_1, data_2)).item()

                # 混合比例
                lam = (1 - alpha) * torch.rand(1).item() + alpha

                cut_ratio = lam ** 4
                cut_ratio = cut_ratio*cut_ratio
                if math.isnan(cut_ratio):
                    cut_ratio = 0.5;

                e.append(cut_ratio)

                cut_area = num_x * num_y * 0.5 + 1
                w = np.random.randint(0, num_x * 0.5 + 1)+1
                h = np.round(cut_area / w)+1

                cut_h = int(w)
                cut_w = int(h)


                # 随机裁剪区域的起始坐标
                cx = torch.randint(0, num_x, (1,))
                cy = torch.randint(0, num_y, (1,))

                # 裁剪区域的坐标
                x0 = max(start_x, start_x + cx - cut_h // 2)
                x1 = min(end_x, start_x + cx + cut_h // 2)
                y0 = max(start_y, start_y + cy - cut_w // 2)
                y1 = min(end_y, start_y + cy + cut_w // 2)


                data[k, :, x0:x1, y0:y1] = data_0[k, :, x0:x1, y0:y1]
                mask1[k, x0:x1, y0:y1] = mask2[k, x0:x1, y0:y1]


    return data, mask1, np.mean(e)

# x = torch.randn(4, 3, 224, 224)
# y = torch.randn(4, 3, 224, 224)
# xx = torch.randn(4, 224, 224)
# yy = torch.randn(4, 224, 224)
# xxx = torch.randn(4, 15, 224, 224)
# yyy = torch.randn(4, 15, 224, 224)
# # 应用cutmix
# data, mask = cutmix(x, x, xx, yy, xxx, yyy, alpha=0.8, p=1,size=2)
# print(data.shape)
# print(mask.shape)