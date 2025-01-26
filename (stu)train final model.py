from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils import count_params, meanIOU, color_map
import cv2
import argparse
from copy import deepcopy
import numpy as np
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
import timeit
import datetime
from utils import cos_matrix,generate_unsup_aug_ds,generate_unsup_aug_sc
import random
from model.farnet import FarNet
from model.CMTFNet import CMTFNet

start = timeit.default_timer()
start_writeable = datetime.datetime.now().strftime('%m-%d_%H-%M')


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed = 4444
set_random_seed(seed)

MODE = None

DATASET = 'MER'  # ['GID-15', 'iSAID', 'MER', 'MSL', 'Vaihingen', 'DFC22]
SPLIT = '1-8'  # ['1-4', '1-8', '100', '300']
GID15_DATASET_PATH = 'Your local path'
iSAID_DATASET_PATH = 'Your local path'
DFC22_DATASET_PATH = 'Your local path'
MER_DATASET_PATH = 'Your local path'
MSL_DATASET_PATH = 'Your local path'
Vaihingen_DATASET_PATH = 'Your local path'


NUM_CLASSES = {'GID-15': 15, 'iSAID': 15, 'DFC22': 12, 'MER': 9, 'MSL': 9, 'Vaihingen': 5}
txt_path = './dataset/splits/' + DATASET
label_path = './cross/masks'


def parse_args():
    parser = argparse.ArgumentParser(description='LSST Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--dataset', type=str, choices=['GID-15', 'iSAID', 'DFC22', 'MER', 'MSL', 'Vaihingen'],
                        default=DATASET)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--crop-size', type=int, default=321)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet101')
    parser.add_argument('--model', type=str,
                        choices=['FarSeg', 'SwinUnet', 'SegFormer', 'UNet', 'LWnet', 'LEDnet', 'FCN', 'ENet', 'SegNext',
                                 'deeplabv3plus', 'pspnet', 'deeplabv2', 'SegNet'],
                        default='deeplabv2')

    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str,
                        default='./dataset/splits/' + DATASET + '/' + SPLIT + '/labeled.txt')
    parser.add_argument('--unlabeled-id-path', type=str,
                        default='./dataset/splits/' + DATASET + '/' + SPLIT + '/unlabeled.txt')
    parser.add_argument('--save-path', type=str,
                        default='./output/' + DATASET + '/' + SPLIT + '_' + '/models_tea')

    args = parser.parse_args()
    return args


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main(args):
    create_path(args.save_path)

    criterion = CrossEntropyLoss(ignore_index=255)

    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, pin_memory=True, num_workers=8,
                           drop_last=False)

    print('\n\n\n================> student train')

    trainset_u = SemiDataset(args.dataset, args.data_root, 'train_stu_u', args.crop_size, train_id_path=txt_path,
                             labeled_id_path=args.labeled_id_path, unlabeled_id_path=args.unlabeled_id_path)

    trainset = SemiDataset(args.dataset, args.data_root, 'train_stu', args.crop_size,
                           args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path, txt_path, label_path)

    trainloader = DataLoader(trainset, batch_size=args.batch_size // 2, shuffle=True,
                             pin_memory=True, num_workers=4, drop_last=True)

    trainloader_u = DataLoader(trainset_u, batch_size=int(args.batch_size // 2), shuffle=True,
                               pin_memory=False, num_workers=4, drop_last=True)

    model, optimizer = init_basic_elems(args)
    print('\nParams: %.1fM' % count_params(model))
    train(model, trainloader, trainloader_u, valloader, criterion, optimizer, args)

    end = timeit.default_timer()
    print('Total time: ' + str(end - start) + ' seconds')


def init_basic_elems(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == 'deeplabv2':
        model = DeepLabV3Plus(args.backbone, NUM_CLASSES[args.dataset]).to(device)
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)


    if args.model == 'FarSeg':
        model = FarNet(NUM_CLASSES[args.dataset])
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    if args.model == 'CMFT':
        model = CMTFNet(classes=NUM_CLASSES[args.dataset])
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)



    model = model.to(device)
    return model, optimizer


def train(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, args):
    iters = 0

    previous_best = 0.0
    previous_best_iou = 0.0

    iters = 0
    total_iters = len(trainloader_u) * args.epochs

    previous_best = 0.0
    previous_best_iou = 0.0

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f  %s" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best, previous_best_iou))

        total_loss, total_loss_l, total_loss_u = 0.0, 0.0, 0.0

        tbar = tqdm(zip(trainloader_l, trainloader_u), total=len(trainloader_u))
        for i, ((img, mask, _), (img_u_w, img_u_s1, img_u_s2)) in enumerate(tbar):

            img_u_w, img = img_u_w.cuda(), img.cuda()
            with torch.no_grad():
                model.eval()
                pred_u_w = model(img_u_w)
                prob_u_w = pred_u_w.softmax(dim=1)
                conf_u_w, mask_u_w = prob_u_w.max(dim=1)


            img, mask = img.cuda(), mask.cuda()
            img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            mask_u_w = mask_u_w.cuda()

            if np.random.uniform(0, 1) < 0.5:
                img_u_w = generate_unsup_aug_ds(img_u_s1, img_u_s2)
                conf_u_w, mask_u_w, img_u_w = generate_unsup_aug_sc(conf_u_w, mask_u_w, img_u_w)


            mask = mask.cuda()
            img = img.cuda()

            model.train()

            num_img = img.shape[0]
            preds = model(torch.cat((img, img_u_w)))
            pred, pred_u_s = preds.split([num_img, num_img])

            prob_u_s = pred_u_s.clone()
            prob_u_ss = 1-cos_matrix(prob_u_s,pred_u_w)
            d = (10.0 / 80.0) ** (epoch / 100.0)
            d = d * 80.0

            em_threshold = np.percentile(prob_u_ss.detach().cpu().numpy().flatten(), d)

            loss_l = criterion(pred, mask)

            pred_u_s, mask_u_w = pred_u_s.cuda(), mask_u_w.cuda()
            loss_u = criterion(pred_u_s, mask_u_w)
            loss_u = loss_u * (prob_u_ss <= em_threshold)
            loss_u = torch.mean(loss_u)

            loss = loss_l + loss_u

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            total_loss += loss.item()
            total_loss_l += loss_l.item()
            total_loss_u += loss_u.item()

            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

            tbar.set_description('Loss: %.3f, Loss_l: %.3f, Loss_u: %.3f' % (
                total_loss / (i + 1), total_loss_l / (i + 1), total_loss_u / (i + 1)))

        if (epoch + 1) % 5 == 0:
            metric = meanIOU(num_classes=NUM_CLASSES[args.dataset])

            model.eval()
            tbar = tqdm(valloader)

            with torch.no_grad():
                for img, mask, _ in tbar:
                    img = img.cuda()
                    pred = model(img)
                    pred = torch.argmax(pred, dim=1)

                    metric.add_batch(pred.cpu().numpy(), mask.numpy())
                    IOU, mIOU = metric.evaluate()

                    tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

            mIOU *= 100.0
            IOU *= 100
            print('IoU: {}  | MIoU: {}'.format(IOU, mIOU))
            if mIOU > previous_best:
                # if previous_best != 0:
                #     os.remove(
                #         os.path.join(args.save_path, '%s_%s_%.2f_tea_.pth' % (args.model, args.backbone, previous_best)))
                previous_best = mIOU
                previous_best_iou = IOU
                torch.save(model.state_dict(),
                           os.path.join(args.save_path, '%s_%.2f.pth' % (args.model, mIOU)))

                best_model = deepcopy(model)

    return best_model


if __name__ == '__main__':
    args = parse_args()
    if args.epochs is None:
        args.epochs = {'GID-15': 100, 'iSAID': 100, 'MER': 100, 'MSL': 100, 'Vaihingen': 100, 'DFC22': 100}[args.dataset]
    if args.lr is None:
        args.lr = 0.001 / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'GID-15': 321, 'iSAID': 321, 'MER': 321, 'MSL': 321, 'Vaihingen': 321, 'DFC22': 321}[
            args.dataset]
    if args.data_root is None:
        args.data_root = {'GID-15': GID15_DATASET_PATH,
                          'iSAID': iSAID_DATASET_PATH,
                          'MER': MER_DATASET_PATH,
                          'MSL': MSL_DATASET_PATH,
                          'Vaihingen': Vaihingen_DATASET_PATH,
                          'DFC22': DFC22_DATASET_PATH}[args.dataset]

    print(args)

    main(args)
