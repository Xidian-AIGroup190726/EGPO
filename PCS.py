from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from dataset_name import *
from utils import count_params, meanIOU, color_map
import torch.nn as nn
import math

import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.CMTFNet import CMTFNet
from model.farnet import FarNet
from test import get_iou
from utils import cos_matrix


loss_save = 0.85
NUM_CLASSES = {'GID-15': 15, 'iSAID': 15, 'DFC22': 12, 'MER': 9, 'MSL': 9, 'Vaihingen': 5}
SPLIT = '1-8'     # ['1-4', '1-8', '100', '300']
DATASET = 'MER'     # ['GID-15', 'iSAID', 'MER', 'MSL', 'Vaihingen', 'DFC22]
WEIGHTS1 = 'weight of model1'
WEIGHTS2 = 'weight of model2'
save_path = './cross/' + DATASET + ' ' + str(loss_save) + ' ' + WEIGHTS1.split('/')[-1].split('_')[0].replace('.pth', '') + ' '  + WEIGHTS2.split('/')[-1].split('_')[0].replace('.pth', '')
labeled_id_path = './dataset/splits/' + DATASET + '/' + SPLIT + '/all.txt'
unlabeled_id_path = './dataset/splits/' + DATASET + '/' + SPLIT + '/unlabeled.txt'

GID15_DATASET_PATH = './dataset/splits/GID-15/'
iSAID_DATASET_PATH = './dataset/splits/iSAID/'
DFC22_DATASET_PATH = './dataset/splits/DFC22/'
MER_DATASET_PATH = 'Your local path'
MSL_DATASET_PATH = 'Your local path'
Vaihingen_DATASET_PATH = 'Your local path'

data_root = {'GID-15': GID15_DATASET_PATH,
              'iSAID': iSAID_DATASET_PATH,
              'MER': MER_DATASET_PATH,
              'MSL': MSL_DATASET_PATH,
              'Vaihingen': Vaihingen_DATASET_PATH,
              'DFC22': DFC22_DATASET_PATH}[DATASET]


if DATASET == 'MSL' or DATASET == 'MER':
    classes, colors = MARS()
elif DATASET == 'iSAID':
    classes, colors = iSAID()
elif DATASET == 'GID-15':
    classes, colors = GID15()
elif DATASET == 'Vaihingen':
    classes, colors = Vaihingen()
elif DATASET == 'DFC22':
    classes, colors = DFC22()


def eval(model, valloader):
    model.eval()
    tbar = tqdm(valloader)

    data_list = []

    with torch.no_grad():
        for img, mask, _ in tbar:
            img = img.cuda()
            pred = model(img)
            pred = torch.argmax(pred, dim=1).cpu().numpy()
            data_list.append([mask.numpy().flatten(), pred.flatten()])


    return get_iou(data_list, NUM_CLASSES[DATASET], None, DATASET)

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)



def criterion(matrix1, matrix2):
    # 确保两个矩阵大小相同
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must be of the same size.")

    # 计算两个矩阵的相同元素的个数
    same_elements = np.sum(matrix1 == matrix2)

    # 计算两个矩阵的总元素个数
    total_elements = matrix1.size

    # 计算相同部分占全部矩阵的比例
    similarity_ratio = same_elements / total_elements

    return similarity_ratio


def predict_and_stitch(img, model, size):
    num_class = NUM_CLASSES[DATASET]

    # 获取原始图像的尺寸
    original_height, original_width = img.shape[-2:]

    num_x = original_height // size
    num_y = original_width // size

    predictions = torch.zeros(img.size(0), num_class, original_height, original_width)

    # 遍历图像的每个块
    for i in range(size):
        for j in range(size):
            # 计算块的坐标
            start_x = j * num_x
            start_y = i * num_y
            end_x = start_x + num_x
            end_y = start_y + num_y

            # 从原图中裁剪出小块
            img_patch = img[:, :, start_y:end_y, start_x:end_x]


            # 预测小块
            with torch.no_grad():  # 不计算梯度
                prediction = model(img_patch)

            # 将预测结果拼接到 predictions 张量中
            predictions[:, :, start_y:end_y, start_x:end_x] = prediction

    return predictions


def sparse_label(pred, ratio,id = None, save_path = None):
    soft_max_output, hard_output = pred.max(dim=1)

    size = soft_max_output.size()

    soft_vector = soft_max_output.view(-1)
    soft_vector = torch.softmax(soft_vector, dim=0)
    soft_max_output = soft_vector.view(size)

    for j in range(soft_max_output.shape[0]):
         soft, hard = soft_max_output[j].cpu().numpy(), hard_output[j].cpu().numpy()
         need = []
         for c in range(NUM_CLASSES[DATASET]):
             soft_clone, hard_clone = deepcopy(soft), deepcopy(hard)
             need.append(ratio_sample(hard_clone, soft_clone, ratio[c], c))
         need = np.min(np.array(need), axis=0)

         #pred = Image.fromarray(need.astype(np.uint8), mode='P')
         #pred.save(save_path + ' masks/' + os.path.basename(id[j].split(' ')[1]))

    return need


def ratio_sample(hard_out, soft_max_out, ratio, s_class):
    single_h = hard_out
    single_s = soft_max_out
    h_index = (single_h != s_class)
    single_h[h_index] = 255
    single_s[h_index] = 0
    all = sorted(soft_max_out[(hard_out == s_class)], reverse=True)
    num = len(all)
    #need_num = int(num * ratio + 0.5)
    need_num = math.ceil(num * ratio)
    if need_num != 0:
        adaptive_threshold = all[need_num - 1]
        mask = (single_s >= adaptive_threshold)
        index = (mask == False)
        single_h[index] = 255
    else:
        single_h[(single_h != 255)] = 255

    return single_h

def main():
    create_path(save_path + ' labels')
    create_path(save_path + ' masks')

    model1 = CMTFNet(classes=NUM_CLASSES[DATASET])
    model1.load_state_dict(torch.load(WEIGHTS1))
    model1 = DataParallel(model1).cuda()

    model2 = FarNet(NUM_CLASSES[DATASET])
    model2.load_state_dict(torch.load(WEIGHTS2))
    model2 = DataParallel(model2).cuda()

    valset = SemiDataset(DATASET, data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=4,
                           shuffle=False, pin_memory=True, num_workers=8, drop_last=False)

    dataset = SemiDataset(DATASET, data_root, 'label', None, None, unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4,
                            drop_last=False)

    model1.eval()
    model2.eval()
    iou_1, ave_1 = eval(model1,valloader)
    iou_2, ave_2 = eval(model2,valloader)
    iou = iou_1 + iou_2

    tbar = tqdm(dataloader)

    with torch.no_grad():
        for img, _, id in tbar:
            img = img.cuda()

            # 概率 类别
            #pred1 = predict_and_stitch(img, model1, size1)
            pred1 = model1(img).cpu()
            #predlabel1 = torch.argmax(pred1, dim=1).cpu().numpy()
            output_1, pred_label_1 = pred1.max(dim=1)

            #pred2 = predict_and_stitch(img, model2, size2)
            pred2 = model2(img).cpu()
            #predlabel2 = torch.argmax(pred2, dim=1).cpu().numpy()
            output_2, pred_label_2 = pred2.max(dim=1)

            loss = cos_matrix(pred1, pred2).cpu().numpy().squeeze(0)


            pred_1 = sparse_label(pred1, iou_1)
            pred_2 = sparse_label(pred2, iou_2)

            for i in range(NUM_CLASSES[DATASET]):
                output_1[(pred_label_1 == i)] = output_1[(pred_label_1 == i)] * iou_1[i]
                output_2[(pred_label_2 == i)] = output_2[(pred_label_2 == i)] * iou_2[i]


            h = (pred1.shape[2])
            w = (pred1.shape[3])
            pred_1_tensor = torch.tensor(pred_1)
            pred_2_tensor = torch.tensor(pred_2)

            # 使用 PyTorch 的比较操作和逻辑索引来避免循环
            pred_tensor = torch.where(output_1 >= output_2, pred_1_tensor, pred_2_tensor)

            # 如果你需要将结果转换回 NumPy 数组，可以使用 .numpy() 方法
            pred = pred_tensor.numpy()
            save = loss < loss_save
            pred = pred.squeeze(0)
            pred[save] = 255


            pred_img = Image.fromarray(pred.astype(np.uint8), mode='P')
            pred_img.save(save_path + ' masks/' + (id[0].split("/")[1]).split(".")[0] + '.png')

            seg_img = np.zeros((int(h), int(w), 3))
            seg_img = torch.tensor(seg_img, dtype=torch.uint8).cpu()

            for c in range(NUM_CLASSES[DATASET]):
                seg_img[:, :, 0] += ((pred[:, :] == c) * colors[c][0]).astype('uint8')
                seg_img[:, :, 1] += ((pred[:, :] == c) * colors[c][1]).astype('uint8')
                seg_img[:, :, 2] += ((pred[:, :] == c) * colors[c][2]).astype('uint8')

            seg_img = seg_img.cpu()  # 确保seg_img在CPU上
            seg_img = Image.fromarray(np.uint8(seg_img))
            seg_img.save(save_path + ' labels/' + (id[0].split("/")[1]).split(".")[0] + '.jpg')




if __name__ == '__main__':

    main()
