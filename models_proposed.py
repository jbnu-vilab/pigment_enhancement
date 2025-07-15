import torch as torch
import torch.nn as nn
from torch.nn import functional as F

import torchvision.models as models

    
class convBlock(nn.Module):
    def __init__(self, input_feature, output_feature, ksize=3, stride=2, pad=1):
        super(convBlock, self).__init__()

        activation = nn.ReLU()


        lists = []
        lists += [nn.Conv2d(input_feature, output_feature, kernel_size=(ksize, ksize), stride=(stride, stride), padding=(pad, pad))]
        lists += [nn.BatchNorm2d(output_feature)]
        lists += [activation]


        self.model = nn.Sequential(*lists)

    def forward(self, x):
        return self.model(x)


class BasicBlockT(nn.Sequential):
    r"""The basic block module (Conv+LeakyReLU[+InstanceNorm]).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm=False):
        body = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1),
            nn.LeakyReLU(0.2)
        ]
        if norm:
            body.append(nn.InstanceNorm2d(out_channels, affine=True))
        super(BasicBlockT, self).__init__(*body)


class TPAMIBackbone(nn.Sequential):
    r"""The 5-layer CNN backbone module in [TPAMI 3D-LUT]
        (https://github.com/HuiZeng/Image-Adaptive-3DLUT).

    Args:
        pretrained (bool, optional): [ignored].
        input_resolution (int, optional): Resolution for pre-downsampling. Default: 256.
        extra_pooling (bool, optional): Whether to insert an extra pooling layer
            at the very end of the module to reduce the number of parameters of
            the subsequent module. Default: False.
    """

    def __init__(self, input_resolution=256, extra_pooling=True):
        body = [
            BasicBlockT(3, 16, stride=2, norm=True),
            BasicBlockT(16, 32, stride=2, norm=True),
            BasicBlockT(32, 64, stride=2, norm=True),
            BasicBlockT(64, 128, stride=2, norm=True),
            BasicBlockT(128, 128, stride=2),
            nn.Dropout(p=0.5),
        ]
        if extra_pooling:
            body.append(nn.AdaptiveAvgPool2d(2))
        super().__init__(*body)
        self.input_resolution = input_resolution
        self.out_channels = 128 * (4 if extra_pooling else 64)

    def forward(self, imgs):
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2,
            mode='bilinear', align_corners=False)
        return super().forward(imgs).view(imgs.shape[0], -1)


class PigNet(nn.Module):
    def __init__(self, config):
        super(PigNet, self).__init__()
        self.control_point_num = config.control_point + 2
        self.feature_num = config.feature_num
        
        self.trans_param = 5.0
        self.offset_param = 0.1

        param_num1 = (self.control_point_num * self.feature_num)
        param_num4 = (3 * self.feature_num)
        param_num2 = (3 * self.feature_num)
        self.classifier = resnet18_224(out_dim=param_num1, out_dim2=param_num2, out_dim4=param_num4, res_size=config.loader_size, res_num=config.res_num, fc_node1=128, fc_node2=128)
            
        self.mid_conv = 2
        conv_list = []
        for i in range(0, self.mid_conv):
            conv_list.append(convBlock(self.feature_num, self.feature_num, ksize=1, stride=1, pad=0))
        if self.mid_conv > 0:
            self.mid_conv_module = nn.Sequential(*conv_list)

        self.colorTransform = colorTransform(self.control_point_num, self.offset_param, config)
        self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=1, stride=1, padding=0, bias=False).cuda()
        self.sigmoid = nn.Sigmoid()


    def forward(self, org_img, color_map_control):
        N, C, H, W = org_img.shape
        self.cls_output = self.classifier(org_img)
        
        cur_idx = 3 * self.feature_num
        transform_params = self.cls_output[:,:cur_idx]
        transform_params = transform_params.reshape(N * self.feature_num, 3)
        transform_params = self.sigmoid(self.trans_param * transform_params)
        epsilon = 1e-10
        t_sum = torch.sum(transform_params, dim=1, keepdim=True)
        transform_params = transform_params / (t_sum + epsilon)
        transform_params = transform_params.reshape(N * self.feature_num, 3, 1, 1)

        org_img = org_img.reshape(1, N * 3, H, W)
        img_f = F.conv2d(input=org_img, weight=transform_params, groups=N)
        img_f = img_f.reshape(N,self.feature_num,H,W)


        plus_idx = self.control_point_num * self.feature_num
        offset_param = self.cls_output[:,cur_idx:cur_idx + plus_idx]
        cur_idx += plus_idx
        img_f_t = self.colorTransform(img_f, offset_param, color_map_control)

        if self.mid_conv > 0:
            img_f_t = self.mid_conv_module(img_f_t)
        
        hyper_params = self.cls_output[:,cur_idx:]
        hyper_params = hyper_params.reshape(N * 3, self.feature_num, 1, 1)
        hyper_params /= self.feature_num

        img_f_t = img_f_t.reshape(1, N * self.feature_num, H, W)
        out_img = F.conv2d(input=img_f_t, weight=hyper_params, groups=N)
        out_img = out_img.reshape(N,3,H,W)

        return out_img

class colorTransform(nn.Module):
    def __init__(self, control_point=16, offset_param=0.04, config=0):
        super(colorTransform, self).__init__()
        self.softmax = nn.Softmax(dim=0)
        self.control_point = control_point
        self.config = config
        self.feature_num = config.feature_num

        self.offset_param = nn.Parameter(torch.tensor([offset_param], dtype=torch.float32))


    def forward(self, org_img, params, color_map_control):
        N, C, H, W = org_img.shape
        color_map_control_x = color_map_control.clone()

        params = params.reshape(N, self.feature_num, self.control_point) * self.offset_param
        color_map_control_y = color_map_control_x + params

        color_map_control_y = torch.cat((color_map_control_y, color_map_control_y[:, :, self.control_point-1:self.control_point]), dim=2)
        color_map_control_x = torch.cat((color_map_control_x, color_map_control_x[:, :, self.control_point-1:self.control_point]), dim=2)
        img_reshaped = org_img.reshape(N, self.feature_num, -1)
        img_reshaped_val = img_reshaped * (self.control_point-1)


        img_reshaped_index = torch.floor(img_reshaped * (self.control_point-1))
        img_reshaped_index = img_reshaped_index.type(torch.int64)
        img_reshaped_index_plus = img_reshaped_index + 1

        img_reshaped_coeff = img_reshaped_val - img_reshaped_index
        img_reshaped_coeff_one = 1.0 - img_reshaped_coeff

        mapped_color_map_control_y = torch.gather(color_map_control_y, 2, img_reshaped_index)
        mapped_color_map_control_y_plus = torch.gather(color_map_control_y, 2, img_reshaped_index_plus)

        out_img_reshaped = img_reshaped_coeff_one * mapped_color_map_control_y + img_reshaped_coeff * mapped_color_map_control_y_plus

        out_img_reshaped = out_img_reshaped.reshape(N, C, H, W)
        return out_img_reshaped



class resnet18_224(nn.Module):

    def __init__(self, out_dim=5, out_dim2=0, out_dim4=0, res_num=18, res_size=224, fc_node1=1024, fc_node2=1024):
        super(resnet18_224, self).__init__()

        self.out_dim2 = out_dim2
        self.out_dim4 = out_dim4
        self.fc_num = 2
        if res_num == 5:
            net = TPAMIBackbone(input_resolution=res_size)
        if res_num == 18:
            net = models.resnet18(pretrained=True)
        elif res_num == 34:
            net = models.resnet34(pretrained=True)

        self.upsample = nn.Upsample(size=(res_size, res_size), mode='bilinear')
        

        net.fc = nn.Identity()

        lists = []
        lists += [nn.Linear(512, fc_node1),
                nn.ReLU(),
                nn.Linear(fc_node1, out_dim)]
        self.fc = nn.Sequential(*lists)

        torch.nn.init.constant_(self.fc[2].weight.data, 0)
        torch.nn.init.constant_(self.fc[2].bias.data, 0)

        if out_dim2 > 0:
            lists = []
            lists += [nn.Linear(512, fc_node2),
                    nn.ReLU(),
                    nn.Linear(fc_node2, out_dim2)]
            self.fc2 = nn.Sequential(*lists)

        if out_dim4 > 0:
            lists = []
            lists += [nn.Linear(512, fc_node2),
                    # nn.BatchNorm2d(1024),
                    nn.ReLU(),
                    nn.Linear(fc_node2, out_dim4)]
            self.fc4 = nn.Sequential(*lists)

            torch.nn.init.constant_(self.fc4[2].weight.data, 0)
            torch.nn.init.constant_(self.fc4[2].bias.data, 1.0 / 64.0)

        self.model = net

    def forward(self, x):
        x = self.upsample(x)
        f = self.model(x)
        f1 = self.fc(f)
        if self.out_dim2 > 0:
            f2 = self.fc2(f)
            f1 = torch.cat((f1, f2), dim=1)
        if self.out_dim4 > 0:
            f4 = self.fc4(f)
            f1 = torch.cat((f1, f4), dim=1)
        return f1