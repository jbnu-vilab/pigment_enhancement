import torch
import torchvision



class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, vgg_mode=0, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        if vgg_mode == 0:
            blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[9:14].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        elif vgg_mode == 1:
            blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[9:19].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[19:26].eval())
        elif vgg_mode == 2:
            blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[23:30].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2], style_layers=[]):
        # input = 0.5 * (input + 1.0)
        # target = 0.5 * (target + 1.0)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
        return loss


class VGGContrastiveLoss(torch.nn.Module):
    def __init__(self, vgg_mode=0, resize=False):
        super(VGGContrastiveLoss, self).__init__()
        blocks = []
        if vgg_mode == 0:
            blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[9:14].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        elif vgg_mode == 1:
            blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[9:19].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[19:26].eval())
        elif vgg_mode == 2:
            blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[23:30].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, pred, target, feature_layers=[0, 1, 2], style_layers=[]):
        # input = 0.5 * (input + 1.0)
        # target = 0.5 * (target + 1.0)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            pred = pred.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        pred = (pred-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
            pred = self.transform(pred, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        z = pred
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            z = block(z)
            if i in feature_layers:
                loss += ( torch.nn.functional.l1_loss(y, z) / torch.nn.functional.l1_loss(x, z) )
        return loss