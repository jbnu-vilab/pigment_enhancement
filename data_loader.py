import torch
import torchvision
import folders
import random
from ppr10k import ImageDataset_paper


class RandomCropWithRandomSize(object):
    def __init__(self):
        self.range = 0
    def __call__(self, sample):
        image = sample
        _, h, w = image.shape
        n_h = random.randint(round(0.6*h), h)
        n_w = random.randint(round(0.6*w), w)
        transforms = torchvision.transforms.RandomCrop((n_h, n_w))
        image = transforms(image)
        return image
# 2. random rotate 0 90 180 270
class RandomRotate90(object):
    def __call__(self, sample):
        image = sample
        deg = random.randint(0, 3)
        image = torch.rot90(image, deg, [1, 2])
        return image
    






class generate_index_image(object):
    def __call__(self, image):
        c,h,w = image.shape
        x = [i for i in range(w)]
        y = [i for i in range(h)]
        x = torch.tensor(x)
        y = torch.tensor(y)
        x, y = torch.meshgrid(x, y, indexing='xy')
        x = x / ((w - 1) / 31.0)
        y = y / ((h - 1) / 31.0)

        xl = torch.floor(x)
        xl[xl < 0] = 0

        xr = torch.ceil(x)
        xr[xr > 31] = 31

        yu = torch.floor(y)
        yu[yu < 0] = 0

        yd = torch.ceil(y)
        yd[yd > 31] = 31

        i1 = 32 * yu + xl
        i2 = 32 * yu + xr

        i3 = 32 * yd + xl
        i4 = 32 * yd + xr

        i1 = torch.reshape(i1, (1, h, w))
        i1 = i1.type(torch.long)
        i2 = torch.reshape(i2, (1, h, w))
        i2 = i2.type(torch.long)
        i3 = torch.reshape(i3, (1, h, w))
        i3 = i3.type(torch.long)
        i4 = torch.reshape(i4, (1, h, w))
        i4 = i4.type(torch.long)

        index_image = torch.cat((i1, i2, i3, i4), dim=0)
        return image, index_image


class generate_color_map(object):
    def __call__(self, image):
        # image, index_image = image

        color_map = torch.zeros(3, 256)
        for i in range(0,256):
            color_map[:,i] = i
        return image, color_map

class DataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, dataset, path, config, batch_size=1, istrain=True, loader_mode=1, num_workers=4):

        self.batch_size = batch_size
        self.istrain = istrain
        self.config = config
        if (dataset == 'adobe5k'):
             # Train transforms
            if istrain:
                transforms = torchvision.transforms.Compose([
                    RandomCropWithRandomSize(), # range [h/2,h] [w/2,w]
                    torchvision.transforms.Resize((config.loader_size, config.loader_size)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    RandomRotate90(),
                    generate_color_map()
                ])


            else:
                transforms = torchvision.transforms.Compose([
                    generate_color_map()
                ])

        if dataset == 'adobe5k':
            self.data = folders.Adobe5kFolder(
                root=path, transform=transforms, istrain=self.istrain)
        
        elif dataset == 'ppr10ka' or dataset == 'ppr10kb' or dataset == 'ppr10kc':
            if dataset == 'ppr10ka':
                retoucher = 'A'
            elif dataset == 'ppr10kb':
                retoucher = 'B'
            elif dataset == 'ppr10kc':
                retoucher = 'C'
            if self.istrain == 1:
                self.data = ImageDataset_paper(root=path, mode="train", use_mask=False, retoucher=retoucher, loader_size=config.loader_size)
            else:
                self.data = ImageDataset_paper(root=path, mode="test", use_mask=False, retoucher=retoucher, loader_size=config.loader_size)
    
        self.num_workers = num_workers
        self.train_sampler = 0



    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        else:
            dataloader = torch.utils.data.DataLoader(self.data, batch_size=1, shuffle=False)
        
        return dataloader