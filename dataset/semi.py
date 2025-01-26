from dataset.transform import crop, hflip, normalize, resize, blur, cutout, color_transformation, geometric_transformation
from copy import deepcopy
import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from utils import generate_unsup_aug_ds


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size, labeled_id_path=None, unlabeled_id_path=None, pseudo_mask_path=None, train_id_path=None, label_path=None, nsample=None):
        """
        :param name: dataset name, GID-15, iSAID, DFC22, MER, MSL, and Vaihingen
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        :param labeled_id_path: path of labeled image ids, needed in train or semi_train mode.
        :param unlabeled_id_path: path of unlabeled image ids, needed in semi_train or label mode.
        :param pseudo_mask_path: path of generated pseudo masks, needed in semi_train mode.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.label_path = label_path
        self.pseudo_mask_path = pseudo_mask_path
        # train_stu_u train_stu train_u train_l
        if mode == 'train_l':
            id_path = labeled_id_path
        if mode == 'train_u' or mode == 'train_stu_u':
            id_path = unlabeled_id_path

        if mode == 'semi_train':
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
            self.ids = \
                self.labeled_ids * math.ceil(len(self.unlabeled_ids) / len(self.labeled_ids)) + self.unlabeled_ids

        elif mode == 'train_stu' or mode == 'train_tea':
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
            self.ids = \
                self.labeled_ids + self.unlabeled_ids
        elif mode == 'train_l' or mode == 'train_u' or mode == 'train_stu_u':
            if mode == 'train_u':
                with open(unlabeled_id_path, 'r') as f:
                    self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                with open(labeled_id_path, 'r') as f:
                    self.ids = f.read().splitlines()
                    self.ids *= math.ceil(nsample / len(self.ids))
                    random.shuffle(self.ids)
                    self.ids = self.ids[:nsample]
            elif mode == 'train_stu_u':
                with open(labeled_id_path, 'r') as f:
                    self.labeled_ids = f.read().splitlines()
                with open(unlabeled_id_path, 'r') as f:
                    self.unlabeled_ids = f.read().splitlines()
                self.ids = \
                    self.labeled_ids + self.unlabeled_ids
                #self.ids = [id for id in self.ids if id not in self.train_id_path]

        else:
            if mode == 'val':
                id_path = 'dataset/splits/%s/val.txt' % name
            elif mode == 'label':
                id_path = unlabeled_id_path
            elif mode == 'train':
                id_path = labeled_id_path
            elif mode == 'mask_iou':
                id_path = train_id_path

            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0]))

        if self.mode == 'val' or self.mode == 'label':
            mask = Image.open(os.path.join(self.root, id.split(' ')[1]))
            img, mask = normalize(img, mask)
            return img, mask, id

        if self.mode == 'mask_iou':
            label = Image.open(os.path.join(self.root, id.split(' ')[1]))
            mask = Image.open(os.path.join(self.label_path, id.split('/')[-1]))
            _, mask = normalize(img, mask)
            _, label = normalize(img, label)
            return mask, label

        if self.mode == 'train' or (self.mode == 'semi_train' and id in self.labeled_ids):
            mask = Image.open(os.path.join(self.root, id.split(' ')[1]))
        elif self.mode == 'train_stu' or self.mode == 'train_tea':
            if id in self.labeled_ids:
                mask = Image.open(os.path.join(self.root, id.split(' ')[1]))
            else:
                mask = Image.open(os.path.join(self.label_path, id.split('/')[-1][:-3]+'png'))


        else:
            mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

        # basic augmentation on all training images
        base_size = 512
        img, mask = resize(img, mask, base_size, (0.5, 2.0))
        img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, p=0.5)

        # strong augmentation on unlabeled images
        if (self.mode == 'train_stu') or (self.mode == 'semi_train' and id in self.unlabeled_ids):

            # img = color_transformation(img)
            img, mask = geometric_transformation(img, mask)
            img, mask = cutout(img, mask, p=0.5)
            img, mask = normalize(img, mask)

            return img, mask, id

        if self.mode == 'train_tea':
            img, mask = geometric_transformation(img, mask)
            img, mask = cutout(img, mask, p=0.5)
            img, mask = normalize(img, mask)

            return img, mask, id

        # strong augmentation on unlabeled images
        if (self.mode == 'train'):
            img_s1, img_s2 = deepcopy(img), deepcopy(img)

            img_s1 = color_transformation(img_s1)
            img_s2 = color_transformation(img_s2)
            #img, mask, img_s1, img_s2 = geometric_transformation(img, mask, img_s1, img_s2)
            #img, mask = cutout(img, mask, p=0.5)
            imgw, mask = normalize(img, mask)

            return imgw, mask, normalize(img_s1), normalize(img_s2)
        elif self.mode == 'train_l':
            #img = color_transformation(img)
            # img, mask = geometric_transformation(img, mask)
            # img, mask = cutout(img, mask, p=0.5)
            return normalize(img, mask)

        else:
            # strong augmentation on unlabeled images

            img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)


            img_s1 = color_transformation(img_s1)
            img_s2 = color_transformation(img_s2)

            return normalize(img_w), normalize(img_s1), normalize(img_s2)




    def __len__(self):
        return len(self.ids)
