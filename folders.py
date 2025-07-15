import torch.utils.data as data
from PIL import Image
import os
import os.path
import torchvision
import torch

class Adobe5kFolder(data.Dataset):

    def __init__(self, root, transform, istrain):
        if istrain == 1:
            org_dir = root + "train/input/"
            gt_dir = root + "train/user-c/"
        else:
            org_dir = root + "test/input/"
            gt_dir = root + "test/user-c/"
        self.org_list = sorted([os.path.join(org_dir, f) for f in os.listdir(org_dir)])
        self.gt_list = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir)])

        self.transform = transform

    def __getitem__(self, index):
        org_name = self.org_list[index]
        gt_name = self.gt_list[index]

        org = pil_loader(org_name)
        gt = pil_loader(gt_name)

        # to tensor
        to_tensor = torchvision.transforms.ToTensor()
        org = to_tensor(org)
        gt = to_tensor(gt)

        output = torch.cat((org,gt), dim=0)


        # concat
        if self.transform is not None:
            output, index1 = self.transform(output)
        org = output[:3]
        gt = output[3:]
        return org, gt, index1, index

    def __len__(self):
        length = len(self.org_list)
        return length


class ppr10kFolder(data.Dataset):

    def __init__(self, root, transform, istrain, jitter, dataset):
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
        self.org_list = sorted([os.path.join(org_dir, f) for f in os.listdir(org_dir)])
        self.gt_list = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir)])

        self.transform = transform
        self.jitter = jitter

    def __getitem__(self, index):
        org_name = self.org_list[index]
        gt_name = self.gt_list[index]

        org = pil_loader(org_name)
        gt = pil_loader(gt_name)

        # to tensor
        to_tensor = torchvision.transforms.ToTensor()
        org = to_tensor(org)
        gt = to_tensor(gt)

        output = torch.cat((org,gt), dim=0)

        # concat
        if self.transform is not None:
            output, index = self.transform(output)
        org = output[:3]
        gt = output[3:]
        return org, gt, index

    def __len__(self):
        length = len(self.org_list)
        return length




def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')