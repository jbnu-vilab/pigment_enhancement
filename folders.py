import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import csv
from openpyxl import load_workbook
import torchvision
import torch
import albumentations as A

class Adobe5kFolder(data.Dataset):

    def __init__(self, root, transform, istrain, jitter):
        #org_dir = os.path.join(root, "/input/")
        #gt_dir = os.path.join(root, "user-c/")
        if istrain == 1:
            org_dir = root + "train/input/"
            gt_dir = root + "train/user-c/"
        else:
            org_dir = root + "test/input/"
            gt_dir = root + "test/user-c/"
        self.org_list = sorted([os.path.join(org_dir, f) for f in os.listdir(org_dir)])
        self.gt_list = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir)])

        self.transform = transform
        self.jitter = jitter

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        org_name = self.org_list[index]
        gt_name = self.gt_list[index]

        org = pil_loader(org_name)
        gt = pil_loader(gt_name)

        # transform_jitter = A.Compose(
        #     [A.transforms.ColorJitter(brightness=0.2, hue=0, contrast=0, saturation=0.2, p=1.0)],
        #     additional_targets={'image0': 'image'}
        # )
        # transformed = transform_jitter(image=org, image0=gt)

        # color jitter
        if self.jitter == 1:
            jitter_trans = torchvision.transforms.ColorJitter(brightness=0.2, saturation=0.2)
            org = jitter_trans(org)
        # to tensor
        to_tensor = torchvision.transforms.ToTensor()
        org = to_tensor(org)
        gt = to_tensor(gt)



        output = torch.cat((org,gt), dim=0)


        # concat

        #sample = self.img[index // self.patch_num].copy()
        if self.transform is not None:
            output, index = self.transform(output)
            #gt = self.transform(gt)
        org = output[:3]
        gt = output[3:]
        return org, gt, index

    def __len__(self):
        length = len(self.org_list)
        return length


class ppr10kFolder(data.Dataset):

    def __init__(self, root, transform, istrain, jitter, dataset):
        #org_dir = os.path.join(root, "/input/")
        #gt_dir = os.path.join(root, "user-c/")
        if istrain == 1:
            org_dir = root + "train/input/"
            if dataset == 'ppr10ka':
                gt_dir = root + "train/target_A/"
            elif dataset == 'ppr10kb':
                gt_dir = root + "train/target_B/"
            elif dataset == 'ppr10kc':
                gt_dir = root + "train/target_C/"
        else:
            org_dir = root + "test/input/"
            if dataset == 'ppr10ka':
                gt_dir = root + "test/target_A/"
            elif dataset == 'ppr10kb':
                gt_dir = root + "test/target_B/"
            elif dataset == 'ppr10kc':
                gt_dir = root + "test/target_C/"
            #gt_dir = root + "test/target_A/"
        self.org_list = sorted([os.path.join(org_dir, f) for f in os.listdir(org_dir)])
        self.gt_list = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir)])

        self.transform = transform
        self.jitter = jitter

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        org_name = self.org_list[index]
        gt_name = self.gt_list[index]

        org = pil_loader(org_name)
        gt = pil_loader(gt_name)

        # transform_jitter = A.Compose(
        #     [A.transforms.ColorJitter(brightness=0.2, hue=0, contrast=0, saturation=0.2, p=1.0)],
        #     additional_targets={'image0': 'image'}
        # )
        # transformed = transform_jitter(image=org, image0=gt)

        # color jitter
        if self.jitter == 1:
            jitter_trans = torchvision.transforms.ColorJitter(brightness=0.2, saturation=0.2)
            org = jitter_trans(org)
        # to tensor
        to_tensor = torchvision.transforms.ToTensor()
        org = to_tensor(org)
        gt = to_tensor(gt)



        output = torch.cat((org,gt), dim=0)


        # concat

        #sample = self.img[index // self.patch_num].copy()
        if self.transform is not None:
            output, index = self.transform(output)
            #gt = self.transform(gt)
        org = output[:3]
        gt = output[3:]
        return org, gt, index

    def __len__(self):
        length = len(self.org_list)
        return length

class UIEBFolder(data.Dataset):

    def __init__(self, root, transform, istrain, jitter):
        #org_dir = os.path.join(root, "/input/")
        #gt_dir = os.path.join(root, "user-c/")
        if istrain == 1:
            org_dir = root + "train/input/"
            gt_dir = root + "train/label/"
        else:
            org_dir = root + "test/input/"
            gt_dir = root + "test/label/"
        self.org_list = sorted([os.path.join(org_dir, f) for f in os.listdir(org_dir)])
        self.gt_list = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir)])

        self.transform = transform
        self.jitter = jitter

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        org_name = self.org_list[index]
        gt_name = self.gt_list[index]

        org = pil_loader(org_name)
        gt = pil_loader(gt_name)

        # color jitter
        if self.jitter == 1:
            jitter_trans = torchvision.transforms.ColorJitter(brightness=0.2, saturation=0.2)
            org = jitter_trans(org)

        # to tensor
        to_tensor = torchvision.transforms.ToTensor()
        org = to_tensor(org)
        gt = to_tensor(gt)
        output = torch.cat((org,gt), dim=0)

        # concat

        #sample = self.img[index // self.patch_num].copy()
        if self.transform is not None:
            output, index = self.transform(output)
            #gt = self.transform(gt)
        org = output[:3]
        gt = output[3:]
        return org, gt, index

    def __len__(self):
        length = len(self.org_list)
        return length

class EUVPFolder(data.Dataset):

    def __init__(self, root, transform, istrain, jitter):
        #org_dir = os.path.join(root, "/input/")
        #gt_dir = os.path.join(root, "user-c/")
        if istrain == 1:
            org_dir = root + "train/input/"
            gt_dir = root + "train/output/"
        else:
            org_dir = root + "test/input/"
            gt_dir = root + "test/output/"
        self.org_list = sorted([os.path.join(org_dir, f) for f in os.listdir(org_dir)])
        self.gt_list = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir)])

        self.transform = transform
        self.jitter = jitter

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        org_name = self.org_list[index]
        gt_name = self.gt_list[index]

        org = pil_loader(org_name)
        gt = pil_loader(gt_name)

        # color jitter
        if self.jitter == 1:
            jitter_trans = torchvision.transforms.ColorJitter(brightness=0.2, saturation=0.2)
            org = jitter_trans(org)

        # to tensor
        to_tensor = torchvision.transforms.ToTensor()
        org = to_tensor(org)
        gt = to_tensor(gt)
        output = torch.cat((org,gt), dim=0)

        # concat

        #sample = self.img[index // self.patch_num].copy()
        if self.transform is not None:
            output, index = self.transform(output)
            #gt = self.transform(gt)
        org = output[:3]
        gt = output[3:]
        return org, gt, index

    def __len__(self):
        length = len(self.org_list)
        return length

class LOLFolder(data.Dataset):

    def __init__(self, root, transform, istrain, jitter):
        #org_dir = os.path.join(root, "/input/")
        #gt_dir = os.path.join(root, "user-c/")
        if istrain == 1:
            org_dir = root + "our485/low/"
            gt_dir = root + "our485/high/"
        else:
            org_dir = root + "eval15/low/"
            gt_dir = root + "eval15/high/"
        self.org_list = sorted([os.path.join(org_dir, f) for f in os.listdir(org_dir)])
        self.gt_list = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir)])

        self.transform = transform
        self.jitter = jitter

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        org_name = self.org_list[index]
        gt_name = self.gt_list[index]

        org = pil_loader(org_name)
        gt = pil_loader(gt_name)

        # color jitter
        if self.jitter == 1:
            jitter_trans = torchvision.transforms.ColorJitter(brightness=0.2, saturation=0.2)
            org = jitter_trans(org)

        # to tensor
        to_tensor = torchvision.transforms.ToTensor()
        org = to_tensor(org)
        gt = to_tensor(gt)
        output = torch.cat((org,gt), dim=0)

        # concat

        #sample = self.img[index // self.patch_num].copy()
        if self.transform is not None:
            output, index = self.transform(output)
            #gt = self.transform(gt)
        org = output[:3]
        gt = output[3:]
        return org, gt, index

    def __len__(self):
        length = len(self.org_list)
        return length


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')