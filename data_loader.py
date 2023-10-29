import torch
import torchvision
import folders
import random
import torchvision.transforms.functional as F
from ppr10k import ImageDataset_paper, ImageDataset_paper2, ImageDataset_paper3
from torch.utils.data.distributed import DistributedSampler

# 1. random crop with random..size
class RandomCropWithRandomSize(object):
    def __init__(self, range1, range2):
        self.range1 = range1
        self.range2 = range2
    def __call__(self, sample):
        image = sample
        # _, h, w = image.size.shape
        n_h = random.randint(self.range1, self.range2)
        n_w = random.randint(self.range1, self.range2)
        #x = tf.image.random_crop(x, [h, w, 6])
        transforms = torchvision.transforms.RandomCrop((n_h, n_w))
        image = transforms(image)
        return image

class RandomCropWithRandomSize2(object):
    def __init__(self, range1, range2):
        self.range1 = range1
        self.range2 = range2
    def __call__(self, sample):
        image = sample
        _, h, w = image.shape
        #n_h = random.randint(h // 2, h)
        #n_w = random.randint(w // 2, w)

        n_h = random.randint(round(0.6*h), h)
        n_w = random.randint(round(0.6*w), w)
        #x = tf.image.random_crop(x, [h, w, 6])
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
    
# 3. normalize * 2 - 1
class norm2_1(object):
    def __init__(self, norm):
        self.norm = norm
    def __call__(self, sample):
        image = sample
        if self.norm == 1:
            image = 2 * image - 1
        return image


class resize_with_4(object):
    def __call__(self, sample):
        image = sample
        _, h, w = image.shape

        if (h % 4) or (w % 4):
            nH = (h // 4) * 4
            nW = (w // 4) * 4
            image = F.resize(image, (nH, nW))
        return image

class resize_with_8(object):
    def __call__(self, sample):
        image = sample
        _, h, w = image.shape

        if (h % 8) or (w % 8):
            nH = (h // 8) * 8
            nW = (w // 8) * 8
            image = F.resize(image, (nH, nW))
        return image


class resize_with_16(object):
    def __call__(self, sample):
        image = sample
        _, h, w = image.shape

        if (h % 16) or (w % 16):
            nH = (h // 16) * 16
            nW = (w // 16) * 16
            image = F.resize(image, (nH, nW))
        return image



class resize_with_div(object):
    def __init__(self, div):
        self.div = div
    def __call__(self, sample):
        if self.div == 1:
            return sample
        else:
            image = sample
            _, h, w = image.shape

            if (h % self.div) or (w % self.div):
                nH = (h // self.div) * self.div
                nW = (w // self.div) * self.div
                image = F.resize(image, (nH, nW))
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
        #x = tf.cast(x, 'float32') / (tf.cast(w - 1, 'float32') / 31.0)
        y = y / ((h - 1) / 31.0)
        #y = tf.cast(y, 'float32') / (tf.cast(h - 1, 'float32') / 31.0)

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
        if (dataset == 'adobe5k' or dataset == 'LOL' or dataset == 'uieb' or dataset == 'hdr'):
             # Train transforms
            if istrain:
                transforms = torchvision.transforms.Compose([
                    RandomCropWithRandomSize2(256, 512), # range [h/2,h] [w/2,w]
                    #torchvision.transforms.Resize((256, 256)),
                    torchvision.transforms.Resize((config.loader_size, config.loader_size)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    RandomRotate90(),
                    norm2_1(config.norm),
                    generate_color_map()
                ])


            else:
                transforms = torchvision.transforms.Compose([
                    norm2_1(config.norm),
                    resize_with_div(config.div),
                    generate_color_map()
                ])
        if (dataset == 'euvp'):
            # Train transforms
            if istrain:
                transforms = torchvision.transforms.Compose([
                    #RandomCropWithRandomSize2(256, 512), # range [h/2,h] [w/2,w]
                    torchvision.transforms.Resize((config.loader_size, config.loader_size)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    RandomRotate90(),
                    #norm2_1(),
                    #generate_index_image(),
                    generate_color_map()
                ])


            else:
                transforms = torchvision.transforms.Compose([
                    #norm2_1(),
                    resize_with_4(),
                    #generate_index_image(),
                    generate_color_map()
                ])

        if dataset == 'adobe5k':
            self.data = folders.Adobe5kFolder(
                root=path, transform=transforms, istrain=self.istrain, jitter=config.jitter)
        elif dataset == 'LOL':
            self.data = folders.LOLFolder(
                root=path, transform=transforms, istrain=self.istrain, jitter=config.jitter)
        elif dataset == 'euvp':
            self.data = folders.EUVPFolder(
                root=path, transform=transforms, istrain=self.istrain, jitter=config.jitter)
        elif dataset == 'uieb':
            self.data = folders.UIEBFolder(
                root=path, transform=transforms, istrain=self.istrain, jitter=config.jitter)
        elif dataset == 'hdr':
            if self.istrain == 1:
                self.data = ImageDataset_paper3(root=path, mode="train", use_mask=False, loader_size=config.loader_size, div=config.div)
            else:
                self.data = ImageDataset_paper3(root=path, mode="test", use_mask=False, loader_size=config.loader_size, div=config.div)
            
        
        elif dataset == 'ppr10ka' or dataset == 'ppr10kb' or dataset == 'ppr10kc':
            if dataset == 'ppr10ka':
                retoucher = 'A'
            elif dataset == 'ppr10kb':
                retoucher = 'B'
            elif dataset == 'ppr10kc':
                retoucher = 'C'
            if self.istrain == 1:
                self.data = ImageDataset_paper(root=path, mode="train", use_mask=False, retoucher=retoucher, loader_size=config.loader_size, div=config.div)
            else:
                self.data = ImageDataset_paper(root=path, mode="test", use_mask=False, retoucher=retoucher, loader_size=config.loader_size, div=config.div)
            #self.data = folders.ppr10kFolder(root=path, transform=transforms, istrain=self.istrain, jitter=config.jitter, dataset=dataset)
        elif dataset == 'adobe5k2':
            if self.istrain == 1:
                self.data = ImageDataset_paper2(root=path, mode="train", use_mask=False)
            else:
                self.data = ImageDataset_paper2(root=path, mode="test", use_mask=False)
        self.num_workers = num_workers
        if self.istrain == 1:
            if self.config.parallel > 0:
                self.train_sampler = DistributedSampler(dataset=self.data, shuffle=True)
            else:
                self.train_sampler = 0
            
        else:
            if self.config.parallel > 0:
                self.test_sampler = DistributedSampler(dataset=self.data, shuffle=False)
            else:
                self.train_sampler = 0



    def get_data(self):
        if self.config.parallel == 0:
            if self.istrain:
                dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            else:
                dataloader = torch.utils.data.DataLoader(self.data, batch_size=1, shuffle=False)
        else:
            if self.istrain:
                
                dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size // self.config.world_size, shuffle=False, num_workers= 16 // self.config.world_size, sampler = self.train_sampler, pin_memory=True )
            else:
                #dataloader = torch.utils.data.DataLoader(self.data, batch_size=1, shuffle=False, sampler=self.test_sampler, pin_memory=True)
                dataloader = torch.utils.data.DataLoader(self.data, batch_size=1, shuffle=False)

        return dataloader