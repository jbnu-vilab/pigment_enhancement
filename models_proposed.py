import torch as torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import torchvision.transforms as T
from vit import ViT, ViT2

import torchvision.models as models
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        r"""
         - inplanes: input channel size
         - planes: output channel size
         - groups, base_width: ResNext나 Wide ResNet의 경우 사용
        """
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            
        # Basic Block의 구조
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # short connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    
class convBlock(nn.Module):
    def __init__(self, input_feature, output_feature, ksize=3, stride=2, pad=1, extra_conv=False, act='silu'):
        super(convBlock, self).__init__()

        if act=='silu':
            activation = nn.SiLU(inplace=True)
        elif act=='gelu':
            activation = nn.GELU()
        elif act=='leaky':
            activation = nn.LeakyReLU(0.1)
        elif act=='relu':
            activation = nn.ReLU()

        lists = []

        if extra_conv:
            lists += [
                nn.Conv2d(input_feature, input_feature, kernel_size=(ksize, ksize), stride=(stride, stride), padding=(pad, pad)),
                nn.BatchNorm2d(input_feature),
                activation,
                nn.Conv2d(input_feature, output_feature, kernel_size=(ksize, ksize), stride=(stride, stride), padding=(pad, pad))
            ]
        else:
            lists += [
                nn.Conv2d(input_feature, output_feature, kernel_size=(ksize, ksize), stride=(stride, stride), padding=(pad, pad)),
                nn.BatchNorm2d(output_feature),
                activation,
            ]

        self.model = nn.Sequential(*lists)

    def forward(self, x):
        return self.model(x)

class convBlock2(nn.Module):
    def __init__(self, input_feature, output_feature, ksize=3, stride=2, pad=1, extra_conv=False, act='silu', bn=True):
        super(convBlock2, self).__init__()
        if act=='silu':
            activation = nn.SiLU(inplace=True)
        elif act=='gelu':
            activation = nn.GELU()
        elif act=='leaky':
            activation = nn.LeakyReLU(0.1)
        elif act=='relu':
            activation = nn.ReLU()
        elif act=='none':
            activation = nn.Identity()

        lists = []
        lists += [nn.Conv2d(input_feature, output_feature, kernel_size=(ksize, ksize), stride=(stride, stride), padding=(pad, pad))]
        if bn == True:
            lists += [nn.BatchNorm2d(output_feature)]
        lists += [activation]
        if extra_conv:
            lists += [nn.Conv2d(output_feature, output_feature, kernel_size=(ksize, ksize), stride=(stride, stride), padding=(pad, pad))]
        
        # if extra_conv:
        #     lists += [
        #         nn.Conv2d(input_feature, output_feature, kernel_size=(ksize, ksize), stride=(stride, stride), padding=(pad, pad)),
        #         nn.BatchNorm2d(output_feature),
        #         activation,  # Swish activation
        #         nn.Conv2d(output_feature, output_feature, kernel_size=(ksize, ksize), stride=(stride, stride), padding=(pad, pad))
        #     ]
        # else:
        #     lists += [
        #         nn.Conv2d(input_feature, output_feature, kernel_size=(ksize, ksize), stride=(stride, stride), padding=(pad, pad)),
        #         nn.BatchNorm2d(output_feature),
        #         activation,
        #     ]

        self.model = nn.Sequential(*lists)

    def forward(self, x):
        return self.model(x)

class resBlock2(nn.Module):
    def __init__(self, input_feature, output_feature, ksize=3, stride=2, pad=1, extra_conv=False, act='silu'):
        super(resBlock2, self).__init__()
        if act=='silu':
            activation = nn.SiLU(inplace=True)
        elif act=='gelu':
            activation = nn.GELU()
        elif act=='leaky':
            activation = nn.LeakyReLU(0.1)
        elif act=='relu':
            activation = nn.ReLU()

        lists = []

        lists += [
            nn.BatchNorm2d(output_feature),
            activation,
            nn.Conv2d(input_feature, output_feature, kernel_size=(ksize, ksize), stride=(stride, stride), padding=(pad, pad)),

            nn.BatchNorm2d(output_feature),
            activation,
            nn.Conv2d(input_feature, output_feature, kernel_size=(ksize, ksize), stride=(stride, stride),
                      padding=(pad, pad)),

        ]

        self.model = nn.Sequential(*lists)

    def forward(self, x):
        return x + self.model(x)

class resBlock3(nn.Module):
    def __init__(self, input_feature, output_feature, ksize=3, stride=2, pad=1, extra_conv=False, act='silu'):
        super(resBlock3, self).__init__()
        if act=='silu':
            activation = nn.SiLU(inplace=True)
        elif act=='gelu':
            activation = nn.GELU()
        elif act=='leaky':
            activation = nn.LeakyReLU(0.1)
        elif act=='relu':
            activation = nn.ReLU()

        lists = []

        lists += [
            nn.BatchNorm2d(output_feature),
            activation,
            nn.Conv2d(input_feature, output_feature, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1)),

            nn.BatchNorm2d(output_feature),
            activation,
            nn.Conv2d(input_feature, output_feature, kernel_size=(1, 1), stride=(stride, stride), padding=(0, 0)),

        ]

        self.model = nn.Sequential(*lists)

    def forward(self, x):
        return x + self.model(x)
    
class colorTransform(nn.Module):
    def __init__(self, control_point=30, use_param=0, trainable_gamma=0, trainable_offset=0, offset_param=0.04, offset_param2=0.04, gamma_param=1.0, config=0):
        super(colorTransform, self).__init__()
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)
        self.control_point = control_point
        self.sigmoid = torch.nn.Sigmoid()
        self.use_param = use_param
        self.config = config

        if trainable_gamma == 1:
            self.gamma_param = nn.Parameter(torch.tensor([gamma_param], dtype=torch.float32))
        else:
            self.gamma_param = gamma_param

        self.epsilon = 1e-8
        if trainable_offset == 1:
            self.offset_param = nn.Parameter(torch.tensor([offset_param], dtype=torch.float32))
            self.offset_param2 = nn.Parameter(torch.tensor([offset_param2], dtype=torch.float32))


        else:
            self.offset_param = offset_param
            self.offset_param2 = offset_param2

    def forward(self, org_img, color_mapping_global, color_mapping_global_a, color_map_control):
        N, C, H, W = org_img.shape
        color_mapping_global_a2 = color_mapping_global_a.clone()


        org_img_reshaped = org_img.reshape(N, 3, -1)
        org_img_reshaped = org_img_reshaped * 255
        org_img_reshaped_l = torch.round(org_img_reshaped).type(torch.int64)

        control_point_position_offset = color_mapping_global[:, :self.control_point * 3] * self.offset_param2



        control_point_position_offset = control_point_position_offset.reshape(N, 3, -1)


        control_point_offset = color_mapping_global[:, self.control_point * 3:6 * self.control_point] * self.offset_param

        # gamma param 1
        control_point_param = (color_mapping_global[:,
                               6 * self.control_point:9 * self.control_point]) * self.gamma_param

        # additional param
        additional_control_point_param = (color_mapping_global[:,
                                          9 * self.control_point:9 * self.control_point + 3]) * self.gamma_param


        additional_control_point_param = additional_control_point_param.reshape(N, 3, -1)
        # additional offset
        additional_control_point_offset = color_mapping_global[:, 9 * self.control_point + 3:9 * self.control_point + 9] * self.offset_param

        additional_control_point_offset = additional_control_point_offset.reshape(N, 3, -1)

        control_point_offset = control_point_offset.reshape(N, 3, -1)
        control_point_param = control_point_param.reshape(N, 3, -1)


        control_point_position = color_map_control[:,:,1:-1] + control_point_position_offset
        control_point_position_s, control_point_position_i = torch.sort(control_point_position, dim=2)
        control_point_offset_s = torch.gather(input=control_point_offset, dim=2, index=control_point_position_i)
        control_point_param_s = torch.gather(input=control_point_param, dim=2, index=control_point_position_i)


        temp1 = torch.zeros(N, 3, 1).cuda(self.config.rank)
        temp2 = torch.ones(N, 3, 1).cuda(self.config.rank)
        control_point_position_s = torch.cat((temp1, control_point_position_s, temp2), dim=2)

        control_point_offset_s = torch.cat((additional_control_point_offset[:, :, 0:1], control_point_offset_s,
                                                additional_control_point_offset[:, :, 1:2]), dim=2)


        control_point_param_s = torch.cat((additional_control_point_param, control_point_param_s), dim=2)


        start_point_x = control_point_position_s[:, :, :-1]
        end_point_x = control_point_position_s[:, :, 1:]

        start_point_y = control_point_position_s[:, :, :-1] + control_point_offset_s[:, :, :-1]
        end_point_y = control_point_position_s[:, :, 1:] + control_point_offset_s[:, :, 1:]

        len_y = end_point_y - start_point_y
        len_x = end_point_x - start_point_x

        start_point_int = torch.ceil(control_point_position_s[:, :, :-1] * 255).type(torch.long)

        start_point_int = start_point_int.unsqueeze(2)
        start_point_int = start_point_int.repeat(1, 1, 256, 1)

        color_map_find = color_mapping_global_a.unsqueeze(3)
        color_map_find = color_map_find.repeat(1, 1, 1, self.control_point + 1)

        color_map_find = torch.where(color_map_find >= start_point_int, 1, 0)
        color_map_find = torch.sum(color_map_find, dim=3)
        color_map_find = color_map_find - 1

        color_mapping_global_a2 = color_mapping_global_a2 / 255
        start_point_x = torch.gather(input=start_point_x, dim=2, index=color_map_find)
        start_point_y = torch.gather(input=start_point_y, dim=2, index=color_map_find)
        len_x = torch.gather(input=len_x, dim=2, index=color_map_find)
        len_y = torch.gather(input=len_y, dim=2, index=color_map_find)
        control_point_param_s = torch.gather(input=control_point_param_s, dim=2, index=color_map_find)

        if self.use_param == 0:
            color_mapping_global_a2 = (color_mapping_global_a2 - start_point_x) / len_x * len_y + start_point_y
        else:
            temp = ((color_mapping_global_a2 - start_point_x) / len_x)
            color_mapping_global_a2 = ((-control_point_param_s) * (temp ** 2) + (
                    control_point_param_s + 1) * temp) * len_y + start_point_y

        index_r_l = org_img_reshaped_l[:, 0]
        index_g_l = org_img_reshaped_l[:, 1]
        index_b_l = org_img_reshaped_l[:, 2]

        color_mapping_r = color_mapping_global_a2[:, 0, :] * 255
        color_mapping_g = color_mapping_global_a2[:, 1, :] * 255
        color_mapping_b = color_mapping_global_a2[:, 2, :] * 255

        color_mapping_r = torch.clamp(color_mapping_r, 0, 255)
        color_mapping_g = torch.clamp(color_mapping_g, 0, 255)
        color_mapping_b = torch.clamp(color_mapping_b, 0, 255)

        mapped_r_u = torch.gather(color_mapping_r, 1, index_r_l)
        mapped_g_u = torch.gather(color_mapping_g, 1, index_g_l)
        mapped_b_u = torch.gather(color_mapping_b, 1, index_b_l)

        mapped_y = torch.cat((mapped_r_u, mapped_g_u, mapped_b_u), dim=1)
        mapped_y = mapped_y.reshape(N, 3, H, W)

        mapped_y = mapped_y / 255
        x = mapped_y

        return x

def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    # if isinstance(m, nn.Conv2d):
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        # init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)



class DCPNet23(nn.Module):
    def __init__(self, config):
        super(DCPNet23, self).__init__()

        self.control_point_num = config.control_point + 2
        self.feature_num = config.feature_num
        self.classifier = resnet18_224(out_dim=self.control_point_num * self.feature_num, res_size=config.res_size, res_num=config.res_num)
        self.params = nn.Parameter(torch.randn(self.feature_num, 3, 1, 1))

        self.colorTransform = colorTransform2(self.control_point_num, config.offset_param, config)

        self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=3, stride=1, padding=1).cuda(config.rank)


        self.sigmoid = nn.Sigmoid()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, org_img, index_image, color_map_control):
        self.cls_output = self.classifier(org_img)
        norm_params = self.sigmoid(self.params)
        epsilon = 1e-10
        w_sum = torch.sum(norm_params, dim=1, keepdim=True)
        norm_params = norm_params / (w_sum + epsilon)

        img_f = F.conv2d(input=org_img, weight=norm_params)

        img_f_t = self.colorTransform(img_f, self.cls_output, index_image, color_map_control)
        #self.conv_emb = F.conv2d(3, self.feature_num, weight=norm_params, kernel_size=1, stride=1, padding=0, bias=False)
        #self.temp_weight =
        #conv_emb = nn.Conv2d(3, self.feature_num, weight= , kernel_size=1, stride=1, padding=0, bias=False)
        out_img = self.conv_out(img_f_t)

        #img_f = self.conv_emb(org_img)

        return out_img


class colorTransform2(nn.Module):
    def __init__(self, control_point=16, offset_param=0.04, config=0):
        super(colorTransform2, self).__init__()
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)
        self.control_point = control_point
        self.sigmoid = torch.nn.Sigmoid()
        self.config = config
        self.feature_num = config.feature_num

        self.epsilon = 1e-8

        self.offset_param = nn.Parameter(torch.tensor([offset_param], dtype=torch.float32))



    def forward(self, org_img, params, color_mapping_global_a, color_map_control):
        N, C, H, W = org_img.shape
        #out_img = torch.zeros_like(org_img)
        color_map_control_x = color_map_control.clone()
        params = params.reshape(N, self.feature_num, self.control_point) * self.offset_param
        color_map_control_y = color_map_control_x + params

        color_map_control_y = torch.cat((color_map_control_y, color_map_control_y[:, :, self.control_point-1:self.control_point]), dim=2)
        color_map_control_x = torch.cat((color_map_control_x, color_map_control_x[:, :, self.control_point-1:self.control_point]), dim=2)
        img_reshaped = org_img.reshape(N, self.feature_num, -1)
        #out_img_reshaped = out_img.reshape(N, self.feature_num, -1)
        img_reshaped_val = img_reshaped * (self.control_point-1)


        img_reshaped_index = torch.floor(img_reshaped * (self.control_point-1))
        img_reshaped_index = img_reshaped_index.type(torch.int64)
        img_reshaped_index_plus = img_reshaped_index + 1

        img_reshaped_coeff = img_reshaped_val - img_reshaped_index
        img_reshaped_coeff_one = 1.0 - img_reshaped_coeff

        mapped_color_map_control_y = torch.gather(color_map_control_y, 2, img_reshaped_index)
        mapped_color_map_control_y_plus = torch.gather(color_map_control_y, 2, img_reshaped_index_plus)

        #out_img_reshaped = (1.0 - img_reshaped_coeff) * mapped_color_map_control_y + img_reshaped_coeff * mapped_color_map_control_y_plus
        out_img_reshaped = img_reshaped_coeff_one * mapped_color_map_control_y + img_reshaped_coeff * mapped_color_map_control_y_plus
        
        # for i in range(0, self.control_point):
        #     mask = img_reshaped_index == i
        #     masked_img_reshaped_coeff = mask * img_reshaped_coeff
        #     masked_img_reshaped_coeff_one = mask * img_reshaped_coeff_one
        #     out_img_reshaped += masked_img_reshaped_coeff_one * color_map_control_y[:,:,i:i+1] + masked_img_reshaped_coeff * color_map_control_y[:,:,i+1:i+2]

        out_img_reshaped = out_img_reshaped.reshape(N, C, H, W)
        return out_img_reshaped


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

    def __init__(self, pretrained=False, input_resolution=256, extra_pooling=True):
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


class DCPNet24(nn.Module):
    def __init__(self, config):
        super(DCPNet24, self).__init__()
        
        self.write_text = config.write_text

        self.control_point_num = config.control_point + 2
        self.feature_num = config.feature_num
        
        self.hyper = config.hyper
        self.xoffset = config.xoffset
        self.transform_num = config.transform_num
        self.conv_num = config.conv_num
        self.last_hyper = config.last_hyper

        self.residual = config.residual
        self.hyper_conv = config.hyper_conv

        self.local_residual = config.local_residual
        self.bias = config.bias
        self.quad = config.quad
        self.trans_param = config.trans_param
        if config.learnable_trans_param == 1:
            self.trans_param = nn.Parameter(torch.tensor([config.trans_param], dtype=torch.float32))

        self.leaky_relu = nn.LeakyReLU(0.1)
        param_num = (self.control_point_num * self.feature_num) * self.transform_num
        param_num1 = param_num
        param_num4 = 0
        if self.bias == 1:
            param_num += (self.feature_num * self.transform_num)
        if self.quad == 1 or self.quad == 4 or self.quad == 5:
            param_num += (self.feature_num * self.transform_num)
        if self.quad == 2 or self.quad == 3 or self.quad == 6 or self.quad == 7:
            param_num += (self.feature_num * self.transform_num) * 2
        if self.last_hyper == 1:
            param_num += (3 * self.feature_num)
            param_num4 += (3 * self.feature_num)
        param_xoffset = 0
        if self.xoffset == 1:
            param_xoffset = ((self.control_point_num - 2) * self.feature_num) * self.transform_num
            param_num += ((self.control_point_num - 2) * self.feature_num) * self.transform_num
        elif self.xoffset == 2:
            param_xoffset = ((self.control_point_num - 1) * self.feature_num) * self.transform_num
            param_num += ((self.control_point_num - 1) * self.feature_num) * self.transform_num

        if self.hyper == 0:
            if config.new_res == 0:
                self.classifier = resnet18_224(out_dim=param_num, res_size=config.res_size, res_num=config.res_num, fc_num=config.fc_num)
            elif config.new_res == 1:
                self.classifier = resnet18_224_2(out_dim=param_num1, out_dim2=0, out_dim3=param_xoffset, out_dim4=param_num4, res_size=config.res_size, res_num=config.res_num,
                                               fc_num=config.fc_num, init_w=config.init_w, init_w2=config.init_w2, init_w_last=config.init_w_last, fc_node=config.fc_node, fc_node1=config.fc_node1, fc_node2=config.fc_node2)
            self.params = nn.Parameter(torch.randn(self.feature_num, 3, 1, 1))
        elif self.hyper == 1:
            if self.hyper_conv == 1:
                param_num += (3 * self.feature_num)
                param_num2 = (3 * self.feature_num)
            elif self.hyper_conv == 3:
                param_num += (3 * self.feature_num) * 9
                param_num2 = (3 * self.feature_num) * 9
            if config.new_res == 0:
                self.classifier = resnet18_224(out_dim=param_num, res_size=config.res_size, res_num=config.res_num, fc_num=config.fc_num)
            elif config.new_res == 1:
                self.classifier = resnet18_224_2(out_dim=param_num1, out_dim2=param_num2, out_dim3=param_xoffset, out_dim4=param_num4, res_size=config.res_size, res_num=config.res_num,
                                                 fc_num=config.fc_num, init_w=config.init_w, init_w2=config.init_w2, init_w_last=config.init_w_last, fc_node=config.fc_node, fc_node1=config.fc_node1, fc_node2=config.fc_node2)
                


        self.mid_conv = config.mid_conv
        conv_list = []
        for i in range(0, self.mid_conv):
            if config.mid_conv_mode == 'res':
                if config.mid_conv_size == 3:
                    conv_list.append(resBlock2(self.feature_num, self.feature_num, ksize=3, stride=1, pad=1, extra_conv=False, act='relu'))
                else:
                    conv_list.append(resBlock2(self.feature_num, self.feature_num, ksize=1, stride=1, pad=0, extra_conv=False, act='relu'))
            elif config.mid_conv_mode == 'res2':
                    conv_list.append(resBlock3(self.feature_num, self.feature_num, ksize=3, stride=1, pad=1, extra_conv=False, act='relu'))
            elif config.mid_conv_mode == 'conv':
                if config.mid_conv_size == 3:
                    ksize1 = 3
                    pad1 = 1
                elif config.mid_conv_size == 1:
                    ksize1 = 1
                    pad1 = 0
                if config.last_relu == 1:
                    act1 = 'relu'
                    bn1 = True
                else:
                    if i < self.mid_conv - 1:
                        act1 = 'relu'
                        bn1 = True
                    else:
                        act1 = 'none'
                        bn1 = False
                conv_list.append(convBlock2(self.feature_num, self.feature_num, ksize=ksize1, stride=1, pad=pad1, extra_conv=False, act=act1, bn=bn1))
            elif config.mid_conv_mode == 'bottleneck1':
                lists = []
                hidden_feature = 256
                
                lists += [nn.Conv2d(self.feature_num, hidden_feature, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))]
                lists += [nn.BatchNorm2d(hidden_feature)]
                lists += [nn.ReLU()]
                mod = nn.Sequential(*lists)
                conv_list.append(mod)
                lists = []
                lists += [nn.Conv2d(hidden_feature, self.feature_num, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))]
                lists += [nn.BatchNorm2d(self.feature_num)]
                lists += [nn.ReLU()]
                mod = nn.Sequential(*lists)
                conv_list.append(mod)
                break
            elif config.mid_conv_mode == 'bottleneck2':
                lists = []
                hidden_feature = 32
                lists += [nn.Conv2d(self.feature_num, hidden_feature, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))]
                lists += [nn.BatchNorm2d(hidden_feature)]
                lists += [nn.ReLU()]
                mod = nn.Sequential(*lists)
                conv_list.append(mod)
                lists = []
                lists += [nn.Conv2d(hidden_feature, self.feature_num, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))]
                lists += [nn.BatchNorm2d(self.feature_num)]
                lists += [nn.ReLU()]
                mod = nn.Sequential(*lists)
                conv_list.append(mod)
                break
                
            elif config.mid_conv_mode == 'conv2':
                act1 = config.act
                ksize1 = 1
                pad1 = 0
                bn1 = True
                conv_list.append(convBlock2(self.feature_num, self.feature_num, ksize=ksize1, stride=1, pad=pad1, extra_conv=False, act=act1, bn=bn1))
            
            elif config.mid_conv_mode == 'conv3':
                lists = []
                lists += [nn.Conv2d(self.feature_num, self.feature_num, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))]
                lists += [nn.InstanceNorm2d(self.feature_num)]
                lists += [nn.ReLU()]
                mod = nn.Sequential(*lists)
                conv_list.append(mod)
                lists = []
                lists += [nn.Conv2d(self.feature_num, self.feature_num, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))]
                lists += [nn.InstanceNorm2d(self.feature_num)]
                lists += [nn.ReLU()]
                mod = nn.Sequential(*lists)
                conv_list.append(mod)
                break
        if self.mid_conv > 0:
            self.mid_conv_module = nn.Sequential(*conv_list)
        if config.xoffset == 0:
            self.colorTransform = colorTransform3(self.control_point_num, config.offset_param, config)
        elif config.xoffset == -1:
            self.colorTransform = colorTransform4(self.control_point_num, config.offset_param, config)
        elif config.xoffset == -2:
            self.colorTransform = colorTransform5(self.control_point_num, config.offset_param, config)
        elif config.xoffset == 1:
            self.colorTransform = colorTransform_xoffset(self.control_point_num, config.offset_param, config.offset_param2, config)
        elif config.xoffset == 2:
            self.colorTransform = colorTransform_xoffset_softmax(self.control_point_num, config.offset_param, config.offset_param2, config)
        elif config.xoffset == -100:
            self.colorTransform = colorTransform_no(self.control_point_num, config.offset_param, config)
        if config.conv_mode == 3:
            self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=3, stride=1, padding=1).cuda(config.rank)
        elif config.conv_mode == 1:
            if config.last_conv_bias == 1:
                bias_flag = True
            else:
                bias_flag = False
            self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=1, stride=1, padding=0, bias=bias_flag).cuda(config.rank)
            if config.last_conv_init == 1:
                torch.nn.init.constant_(self.conv_out.weight.data, 1.0 / self.feature_num)
                if bias_flag:
                    torch.nn.init.constant_(self.conv_out.bias.data, 0)

        self.act = 'relu'
        self.pixelwise_multi = config.pixelwise_multi
        if self.pixelwise_multi == 1 or self.pixelwise_multi == 3 or self.pixelwise_multi == 4 or self.pixelwise_multi == 6:
            self.resize = T.Resize((config.local_size, config.local_size))
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

            self.res5 = convBlock2(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)  # s5

            self.res6 = convBlock2(input_feature=256, output_feature=256, ksize=3, stride=1, pad=1, act=self.act)  # s5

            # upsample and concat
            self.res7 = convBlock2(input_feature=384, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

            # upsample and concat
            self.res8 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

             # upsample and concat
            self.res9 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

            # upsample and concat
            self.res10 = convBlock2(input_feature=48, output_feature=self.feature_num, ksize=3, stride=1, pad=1, act=self.act)  # s1

            self.res11 = nn.Conv2d(self.feature_num, self.feature_num, kernel_size=3, padding=1)
        
        if self.pixelwise_multi == 2 or self.pixelwise_multi == 5:
            self.resize = T.Resize((config.local_size, config.local_size))
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

            self.res5 = convBlock2(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)  # s5

            self.res6 = convBlock2(input_feature=256, output_feature=256, ksize=3, stride=1, pad=1, act=self.act)  # s5

            # upsample and concat
            self.res7 = convBlock2(input_feature=384, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

            # upsample and concat
            self.res8 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

             # upsample and concat
            self.res9 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

            # upsample and concat
            self.res10 = convBlock2(input_feature=48, output_feature=self.feature_num * 2, ksize=3, stride=1, pad=1, act=self.act)  # s1

            self.res11 = nn.Conv2d(self.feature_num * 2, self.feature_num * 2, kernel_size=3, padding=1)

        self.div = config.div
        if self.local_residual == 1:
            if self.div == 4:
                self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
                self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
                self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

                self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
                # upsample and concat
                self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
                # upsample and concat
                self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
                self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True, act=self.act)
            elif self.div == 16:
                self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
                self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
                self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

                self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

                self.res5 = convBlock2(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)  # s5

                self.res6 = convBlock2(input_feature=256, output_feature=256, ksize=3, stride=1, pad=1, act=self.act)  # s5

                # upsample and concat
                self.res7 = convBlock2(input_feature=384, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

                # upsample and concat
                self.res8 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

                 # upsample and concat
                self.res9 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

                # upsample and concat
                self.res10 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1

                self.res11 = nn.Conv2d(16, 3, kernel_size=3, padding=1)

            elif self.div == 8:
                self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
                self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
                self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

                self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

                self.res5 = convBlock2(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

                # upsample and concat
                self.res6 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

                 # upsample and concat
                self.res7 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

                # upsample and concat
                self.res8 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1

                self.res9 = nn.Conv2d(16, 3, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.act_mode = config.act_mode

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, org_img, index_image, color_map_control):
        N, C, H, W = org_img.shape
        self.cls_output = self.classifier(org_img)
        if self.pixelwise_multi >= 1:
            y = self.resize(org_img)
            y1 = self.res1(y)
            y2 = self.res2(y1)
            y3 = self.res3(y2)
            y4 = self.res4(y3)
            y5 = self.res5(y4)
            y5 = self.res6(y5)
            y4 = self.res7(torch.cat((y4, F.upsample_bilinear(y5, scale_factor=2)), dim=1))
            y3 = self.res8(torch.cat((y3, F.upsample_bilinear(y4, scale_factor=2)), dim=1))
            y2 = self.res9(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
            y1 = self.res10(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
            y1 = self.res11(y1)
            m = T.Resize((H, W))
            y = m(y1)
        if self.hyper == 0:
            if self.act_mode == 'sigmoid':
                norm_params = self.sigmoid(self.trans_param * self.params)
                epsilon = 1e-10
                w_sum = torch.sum(torch.abs(norm_params), dim=1, keepdim=True)
                norm_params = norm_params / (w_sum + epsilon)
            elif self.act_mode == 'tanh':
                norm_params = self.tanh(self.params)
                epsilon = 1e-10
                w_norm = torch.norm(norm_params, dim=1, keepdim=True)
                norm_params = norm_params / (w_norm + epsilon)

            # 64 x 3 x 1 x 1
            img_f = F.conv2d(input=org_img, weight=norm_params)
            if self.act_mode == 'tanh':
                img_f = (img_f + 1) / 2
            cur_idx = self.control_point_num * self.feature_num
            img_f_t = self.colorTransform(img_f, self.cls_output[:,:cur_idx], index_image, color_map_control)
            if self.act_mode == 'tanh':
                img_f_t = img_f_t * 2 -1
        elif self.hyper == 1:
            N, C, H, W = org_img.shape
            if self.hyper_conv == 1:
                cur_idx = 3 * self.feature_num
            elif self.hyper_conv == 3:
                cur_idx = 3 * self.feature_num * 9
            transform_params = self.cls_output[:,:cur_idx]
            if self.bias == 1:
                bias_params = self.cls_output[:,cur_idx:cur_idx + self.feature_num]
                cur_idx += self.feature_num
            if self.hyper_conv == 1:
                transform_params = transform_params.reshape(N * self.feature_num, 3)
            elif self.hyper_conv == 3:
                transform_params = transform_params.reshape(N * self.feature_num, 27)
            if self.act_mode == 'sigmoid':
                transform_params = self.sigmoid(self.trans_param * transform_params)
                epsilon = 1e-10
                t_sum = torch.sum(transform_params, dim=1, keepdim=True)
                transform_params = transform_params / (t_sum + epsilon)
            elif self.act_mode == 'tanh':
                transform_params = self.tanh(transform_params)
                epsilon = 1e-10
                w_norm = torch.sum(torch.abs(transform_params), dim=1, keepdim=True)
                transform_params = transform_params / (w_norm + epsilon)
            elif self.act_mode == 'minmax' or self.act_mode == 'trunc' or self.act_mode == 'minmax2':
                transform_params = transform_params

            if self.hyper_conv == 1:
                transform_params = transform_params.reshape(N * self.feature_num, 3, 1, 1)
            elif self.hyper_conv == 3:
                transform_params = transform_params.reshape(N * self.feature_num, 3, 3, 3)
            #transform_params = transform_params.permute(1,2,0,3,4)
            #org_img = org_img.permute(1,0,2,3)
            org_img = org_img.reshape(1, N * 3, H, W)
            if self.bias == 1:
                bias_params = bias_params.reshape(N * self.feature_num)
                if self.hyper_conv == 1:
                    img_f = F.conv2d(input=org_img, weight=transform_params, bias=bias_params, groups=N)
                elif self.hyper_conv == 3:
                    img_f = F.conv2d(input=org_img, weight=transform_params, bias=bias_params, groups=N, padding=1)
            else:
                if self.hyper_conv == 1:
                    img_f = F.conv2d(input=org_img, weight=transform_params, groups=N)
                elif self.hyper_conv == 3:
                    img_f = F.conv2d(input=org_img, weight=transform_params, groups=N, padding=1)

            img_f = img_f.reshape(N,self.feature_num,H,W)

            if self.quad == 1:
                quad_params = self.tanh(self.cls_output[:,cur_idx:cur_idx + self.feature_num])
                cur_idx += self.feature_num

                quad_params = quad_params.reshape(N, self.feature_num, 1, 1)
                img_f = img_f + quad_params * img_f * (1.0 - img_f)
            
            elif self.quad == 6 or self.quad == 7:
                gamma_params = self.cls_output[:,cur_idx:cur_idx + self.feature_num]
                cur_idx += self.feature_num
                beta_params = self.cls_output[:,cur_idx:cur_idx + self.feature_num]
                cur_idx += self.feature_num

                gamma_params = gamma_params.reshape(N, self.feature_num, 1, 1)
                beta_params = beta_params.reshape(N, self.feature_num, 1, 1)
                if self.quad == 6:
                    img_f = img_f * gamma_params + beta_params
                elif self.quad == 7:
                    img_f = img_f * gamma_params

            if self.pixelwise_multi == 1:
                img_f = img_f * y
            elif self.pixelwise_multi == 2:
                img_f = img_f * y[:,:self.feature_num,:,:] + y[:,self.feature_num:,:,:]
            elif self.pixelwise_multi == 3:
                y = self.tanh(y)
                img_f = img_f + y * img_f * ( 1.0 - img_f)
            if self.act_mode == 'minmax' or self.act_mode == 'minmax2':
                    # normalize
                min_val, _ = img_f.min(dim=2, keepdim=True)
                min_val, _ = min_val.min(dim=3, keepdim=True)

                max_val, _ = img_f.max(dim=2, keepdim=True)
                max_val, _ = max_val.max(dim=3, keepdim=True)
                img_f = (img_f - min_val) / (max_val - min_val)
            elif self.act_mode == 'trunc':
                img_f = torch.clamp(img_f, 0, 1)

            if self.act_mode == 'tanh':
                img_f = (img_f + 1.0) / 2
                img_f = torch.clamp(img_f, 0, 1)

            for i in range(0,self.transform_num):
                plus_idx = self.control_point_num * self.feature_num
                if self.xoffset == 1:
                    plus_idx += ((self.control_point_num - 2) * self.feature_num)
                elif self.xoffset == 2:
                    plus_idx += ((self.control_point_num - 1) * self.feature_num)
                offset_param = self.cls_output[:,cur_idx:cur_idx + plus_idx]
                cur_idx += plus_idx
                img_f_t = self.colorTransform(img_f, offset_param, index_image, color_map_control)
                if self.act_mode == 'minmax2':
                    img_f_t = min_val + (max_val - min_val) * img_f_t
                if i < self.transform_num - 1:
                    # normalize
                    min_val, _ = img_f_t.min(dim=2, keepdim=True)
                    min_val, _ = min_val.min(dim=3, keepdim=True)

                    max_val, _ = img_f_t.max(dim=2, keepdim=True)
                    max_val, _ = max_val.max(dim=3, keepdim=True)
                    img_f_t = (img_f_t - min_val) / (max_val - min_val)
                    if self.act_mode == 'tanh':
                        img_f_t = img_f_t * 2.0 - 1

        if self.pixelwise_multi == 4:
            img_f_t = img_f_t * y
        elif self.pixelwise_multi == 5:
            img_f_t = img_f_t * y[:,:self.feature_num,:,:] + y[:,self.feature_num:,:,:]
        elif self.pixelwise_multi == 6:
            y = self.tanh(y)
            img_f_t = img_f_t + y * img_f_t * ( 1.0 - img_f_t)

        if self.mid_conv > 0:
            img_f_t = self.mid_conv_module(img_f_t)
        if self.last_hyper == 1:
            hyper_params = self.cls_output[:,cur_idx:]
            hyper_params = hyper_params.reshape(N * 3, self.feature_num, 1, 1)
            #hyper_params = self.tanh(hyper_params)
            hyper_params /= self.feature_num

            img_f_t = img_f_t.reshape(1, N * self.feature_num, H, W)
            out_img = F.conv2d(input=img_f_t, weight=hyper_params, groups=N)
            out_img = out_img.reshape(N,3,H,W)
        elif self.last_hyper == 0:
            out_img = self.conv_out(img_f_t)
        elif self.last_hyper == 2:
            transform_params = transform_params.reshape(N, self.feature_num, 3)
            inverse_params = torch.linalg.pinv(transform_params)
            #z = torch.matmul(inverse_params, transform_params)
            inverse_params = inverse_params.reshape(N * 3, self.feature_num, 1, 1)
            img_f_t = img_f_t.reshape(1, N * self.feature_num, H, W)
            out_img = F.conv2d(input=img_f_t, weight=inverse_params, groups=N)
            out_img = out_img.reshape(N,3,H,W)

        if self.quad == 2 or self.quad == 3:
            gamma_params = self.cls_output[:,cur_idx:cur_idx + self.feature_num]
            cur_idx += self.feature_num
            beta_params = self.cls_output[:,cur_idx:cur_idx + self.feature_num]
            cur_idx += self.feature_num

            gamma_params = gamma_params.reshape(N, self.feature_num, 1, 1)
            beta_params = beta_params.reshape(N, self.feature_num, 1, 1)
            img_f_t = img_f_t * gamma_params + beta_params
            if self.quad == 3:
                img_f_t = self.relu(img_f_t)
        elif self.quad == 4:
            quad_params = self.tanh(self.cls_output[:,cur_idx:cur_idx + self.feature_num])
            cur_idx += self.feature_num
            quad_params = quad_params.reshape(N, self.feature_num, 1, 1)
            img_f_t = img_f_t + quad_params * img_f_t * (1.0 - img_f_t)
        elif self.quad == 5:
            gamma_params = self.sigmoid(self.cls_output[:,cur_idx:cur_idx + self.feature_num])
            cur_idx += self.feature_num 
            gamma_params = gamma_params.reshape(N, self.feature_num, 1, 1)
            img_f_t = img_f_t * gamma_params


        #img_f = self.conv_emb(org_img)
        if self.residual == 1:
            org_img = org_img.reshape(N, 3, H, W)
            out_img = out_img + org_img

        if self.local_residual == 1:
            if self.div == 4:
                y1 = self.res1(out_img)
                y2 = self.res2(y1)
                y3 = self.res3(y2)
                y3 = self.res4(y3)
                y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
                y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
                y1 = self.res7(y1)
            elif self.div == 16:
                y1 = self.res1(out_img)
                y2 = self.res2(y1)
                y3 = self.res3(y2)
                y4 = self.res4(y3)
                y5 = self.res5(y4)
                y5 = self.res6(y5)
                y4 = self.res7(torch.cat((y4, F.upsample_bilinear(y5, scale_factor=2)), dim=1))
                y3 = self.res8(torch.cat((y3, F.upsample_bilinear(y4, scale_factor=2)), dim=1))
                y2 = self.res9(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
                y1 = self.res10(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
                y1 = self.res11(y1)
            elif self.div == 8:
                y1 = self.res1(out_img)
                y2 = self.res2(y1)
                y3 = self.res3(y2)
                y4 = self.res4(y3)
                y4 = self.res5(y4)
                y3 = self.res6(torch.cat((y3, F.upsample_bilinear(y4, scale_factor=2)), dim=1))
                y2 = self.res7(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
                y1 = self.res8(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
                y1 = self.res9(y1)

            out_img = y1 + out_img
        if self.write_text == 0:
            return out_img
        else:
            return out_img, transform_params, offset_param, hyper_params


class DCPNet24_4K(nn.Module):
    def __init__(self, config):
        super(DCPNet24_4K, self).__init__()
        
        self.write_text = config.write_text

        self.control_point_num = config.control_point + 2
        self.feature_num = config.feature_num
        
        self.hyper = config.hyper
        self.xoffset = config.xoffset
        self.transform_num = config.transform_num
        self.conv_num = config.conv_num
        self.last_hyper = config.last_hyper

        self.residual = config.residual
        self.hyper_conv = config.hyper_conv

        self.local_residual = config.local_residual
        self.bias = config.bias
        self.quad = config.quad
        self.trans_param = config.trans_param
        if config.learnable_trans_param == 1:
            self.trans_param = nn.Parameter(torch.tensor([config.trans_param], dtype=torch.float32))

        self.leaky_relu = nn.LeakyReLU(0.1)
        param_num = (self.control_point_num * self.feature_num) * self.transform_num
        param_num1 = param_num
        param_num4 = 0
        if self.bias == 1:
            param_num += (self.feature_num * self.transform_num)
        if self.quad == 1 or self.quad == 4 or self.quad == 5:
            param_num += (self.feature_num * self.transform_num)
        if self.quad == 2 or self.quad == 3 or self.quad == 6 or self.quad == 7:
            param_num += (self.feature_num * self.transform_num) * 2
        if self.last_hyper == 1:
            param_num += (3 * self.feature_num)
            param_num4 += (3 * self.feature_num)
        param_xoffset = 0
        if self.xoffset == 1:
            param_xoffset = ((self.control_point_num - 2) * self.feature_num) * self.transform_num
            param_num += ((self.control_point_num - 2) * self.feature_num) * self.transform_num
        elif self.xoffset == 2:
            param_xoffset = ((self.control_point_num - 1) * self.feature_num) * self.transform_num
            param_num += ((self.control_point_num - 1) * self.feature_num) * self.transform_num

        if self.hyper == 0:
            if config.new_res == 0:
                self.classifier = resnet18_224(out_dim=param_num, res_size=config.res_size, res_num=config.res_num, fc_num=config.fc_num)
            elif config.new_res == 1:
                self.classifier = resnet18_224_2(out_dim=param_num, out_dim2=0, out_dim3=param_xoffset, out_dim4=param_num4, res_size=config.res_size, res_num=config.res_num,
                                               fc_num=config.fc_num, init_w=config.init_w, init_w2=config.init_w2, init_w_last=config.init_w_last, fc_node=config.fc_node, fc_node1=config.fc_node1, fc_node2=config.fc_node2)
            self.params = nn.Parameter(torch.randn(self.feature_num, 3, 1, 1))
        elif self.hyper == 1:
            if self.hyper_conv == 1:
                param_num += (3 * self.feature_num)
                param_num2 = (3 * self.feature_num)
            elif self.hyper_conv == 3:
                param_num += (3 * self.feature_num) * 9
                param_num2 = (3 * self.feature_num) * 9
            if config.new_res == 0:
                self.classifier = resnet18_224(out_dim=param_num, res_size=config.res_size, res_num=config.res_num, fc_num=config.fc_num)
            elif config.new_res == 1:
                self.classifier = resnet18_224_2(out_dim=param_num1, out_dim2=param_num2, out_dim3=param_xoffset, out_dim4=param_num4, res_size=config.res_size, res_num=config.res_num,
                                                 fc_num=config.fc_num, init_w=config.init_w, init_w2=config.init_w2, init_w_last=config.init_w_last, fc_node=config.fc_node, fc_node1=config.fc_node1, fc_node2=config.fc_node2)
                


        self.mid_conv = config.mid_conv
        conv_list = []
        for i in range(0, self.mid_conv):
            if config.mid_conv_mode == 'res':
                if config.mid_conv_size == 3:
                    conv_list.append(resBlock2(self.feature_num, self.feature_num, ksize=3, stride=1, pad=1, extra_conv=False, act='relu'))
                else:
                    conv_list.append(resBlock2(self.feature_num, self.feature_num, ksize=1, stride=1, pad=0, extra_conv=False, act='relu'))
            elif config.mid_conv_mode == 'res2':
                    conv_list.append(resBlock3(self.feature_num, self.feature_num, ksize=3, stride=1, pad=1, extra_conv=False, act='relu'))
            else:
                if config.mid_conv_size == 3:
                    ksize1 = 3
                    pad1 = 1
                elif config.mid_conv_size == 1:
                    ksize1 = 1
                    pad1 = 0
                if config.last_relu == 1:
                    act1 = 'relu'
                    bn1 = True
                else:
                    if i < self.mid_conv - 1:
                        act1 = 'relu'
                        bn1 = True
                    else:
                        act1 = 'none'
                        bn1 = False
                conv_list.append(convBlock2(self.feature_num, self.feature_num, ksize=ksize1, stride=1, pad=pad1, extra_conv=False, act=act1, bn=bn1))
        if self.mid_conv > 0:
            self.mid_conv_module = nn.Sequential(*conv_list)
        if config.xoffset == 0:
            self.colorTransform = colorTransform31(self.control_point_num, config.offset_param, config)
        elif config.xoffset == -1:
            self.colorTransform = colorTransform4(self.control_point_num, config.offset_param, config)
        elif config.xoffset == -2:
            self.colorTransform = colorTransform5(self.control_point_num, config.offset_param, config)
        elif config.xoffset == 1:
            self.colorTransform = colorTransform_xoffset(self.control_point_num, config.offset_param, config.offset_param2, config)
        elif config.xoffset == 2:
            self.colorTransform = colorTransform_xoffset_softmax(self.control_point_num, config.offset_param, config.offset_param2, config)
        if config.conv_mode == 3:
            self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=3, stride=1, padding=1).cuda(config.rank)
        elif config.conv_mode == 1:
            if config.last_conv_bias == 1:
                bias_flag = True
            else:
                bias_flag = False
            self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=1, stride=1, padding=0, bias=bias_flag).cuda(config.rank)
            if config.last_conv_init == 1:
                torch.nn.init.constant_(self.conv_out.weight.data, 1.0 / self.feature_num)
                if bias_flag:
                    torch.nn.init.constant_(self.conv_out.bias.data, 0)

        self.act = 'relu'
        self.pixelwise_multi = config.pixelwise_multi
        if self.pixelwise_multi == 1 or self.pixelwise_multi == 3 or self.pixelwise_multi == 4 or self.pixelwise_multi == 6:
            self.resize = T.Resize((config.local_size, config.local_size))
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

            self.res5 = convBlock2(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)  # s5

            self.res6 = convBlock2(input_feature=256, output_feature=256, ksize=3, stride=1, pad=1, act=self.act)  # s5

            # upsample and concat
            self.res7 = convBlock2(input_feature=384, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

            # upsample and concat
            self.res8 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

             # upsample and concat
            self.res9 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

            # upsample and concat
            self.res10 = convBlock2(input_feature=48, output_feature=self.feature_num, ksize=3, stride=1, pad=1, act=self.act)  # s1

            self.res11 = nn.Conv2d(self.feature_num, self.feature_num, kernel_size=3, padding=1)
        
        if self.pixelwise_multi == 2 or self.pixelwise_multi == 5:
            self.resize = T.Resize((config.local_size, config.local_size))
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

            self.res5 = convBlock2(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)  # s5

            self.res6 = convBlock2(input_feature=256, output_feature=256, ksize=3, stride=1, pad=1, act=self.act)  # s5

            # upsample and concat
            self.res7 = convBlock2(input_feature=384, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

            # upsample and concat
            self.res8 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

             # upsample and concat
            self.res9 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

            # upsample and concat
            self.res10 = convBlock2(input_feature=48, output_feature=self.feature_num * 2, ksize=3, stride=1, pad=1, act=self.act)  # s1

            self.res11 = nn.Conv2d(self.feature_num * 2, self.feature_num * 2, kernel_size=3, padding=1)

        self.div = config.div
        if self.local_residual == 1:
            if self.div == 4:
                self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
                self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
                self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

                self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
                # upsample and concat
                self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
                # upsample and concat
                self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
                self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True, act=self.act)
            elif self.div == 16:
                self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
                self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
                self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

                self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

                self.res5 = convBlock2(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)  # s5

                self.res6 = convBlock2(input_feature=256, output_feature=256, ksize=3, stride=1, pad=1, act=self.act)  # s5

                # upsample and concat
                self.res7 = convBlock2(input_feature=384, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

                # upsample and concat
                self.res8 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

                 # upsample and concat
                self.res9 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

                # upsample and concat
                self.res10 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1

                self.res11 = nn.Conv2d(16, 3, kernel_size=3, padding=1)

            elif self.div == 8:
                self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
                self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
                self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

                self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

                self.res5 = convBlock2(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

                # upsample and concat
                self.res6 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

                 # upsample and concat
                self.res7 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

                # upsample and concat
                self.res8 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1

                self.res9 = nn.Conv2d(16, 3, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.act_mode = config.act_mode

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, org_img, index_image, color_map_control, is_train):
        N, C, H, W = org_img.shape
        self.cls_output = self.classifier(org_img)
        if is_train == 1:
            if self.hyper == 1:
                N, C, H, W = org_img.shape
                if self.hyper_conv == 1:
                    cur_idx = 3 * self.feature_num
                transform_params = self.cls_output[:,:cur_idx]
                if self.hyper_conv == 1:
                    transform_params = transform_params.reshape(N * self.feature_num, 3)
                if self.act_mode == 'sigmoid':
                    transform_params = self.sigmoid(self.trans_param * transform_params)
                    epsilon = 1e-10
                    t_sum = torch.sum(transform_params, dim=1, keepdim=True)
                    transform_params = transform_params / (t_sum + epsilon)
                
                if self.hyper_conv == 1:
                    transform_params = transform_params.reshape(N * self.feature_num, 3, 1, 1)
            
                org_img = org_img.reshape(1, N * 3, H, W)
            
                if self.hyper_conv == 1:
                    org_img = F.conv2d(input=org_img, weight=transform_params, groups=N)

                org_img = org_img.reshape(N,self.feature_num,H,W)

                
                for i in range(0,self.transform_num):
                    plus_idx = self.control_point_num * self.feature_num
                    if self.xoffset == 1:
                        plus_idx += ((self.control_point_num - 2) * self.feature_num)
                    elif self.xoffset == 2:
                        plus_idx += ((self.control_point_num - 1) * self.feature_num)
                    offset_param = self.cls_output[:,cur_idx:cur_idx + plus_idx]
                    cur_idx += plus_idx
                    org_img = self.colorTransform(org_img, offset_param, index_image, color_map_control)

            if self.mid_conv > 0:
                org_img = self.mid_conv_module(org_img)
            if self.last_hyper == 1:
                hyper_params = self.cls_output[:,cur_idx:]
                hyper_params = hyper_params.reshape(N * 3, self.feature_num, 1, 1)
                hyper_params /= self.feature_num

                org_img = org_img.reshape(1, N * self.feature_num, H, W)
                org_img = F.conv2d(input=org_img, weight=hyper_params, groups=N)
                org_img = org_img.reshape(N,3,H,W)


            if self.write_text == 0:
                return org_img
            else:
                return org_img, transform_params, offset_param, hyper_params
        elif is_train == 0:
            div_img = 2
            N2, C2, H2, W2 = org_img.shape
            org_img2 = org_img.clone().detach()
            if self.hyper_conv == 1:
                cur_idx = 3 * self.feature_num
            transform_params = self.cls_output[:,:cur_idx]
            if self.hyper_conv == 1:
                transform_params = transform_params.reshape(N * self.feature_num, 3)
            if self.act_mode == 'sigmoid':
                transform_params = self.sigmoid(self.trans_param * transform_params)
                epsilon = 1e-10
                t_sum = torch.sum(transform_params, dim=1, keepdim=True)
                transform_params = transform_params / (t_sum + epsilon)  
                if self.hyper_conv == 1:
                    transform_params = transform_params.reshape(N * self.feature_num, 3, 1, 1)
            plus_idx = self.control_point_num * self.feature_num
            if self.xoffset == 1:
                plus_idx += ((self.control_point_num - 2) * self.feature_num)
            elif self.xoffset == 2:
                plus_idx += ((self.control_point_num - 1) * self.feature_num)
            offset_param = self.cls_output[:,cur_idx:cur_idx + plus_idx]
            cur_idx += plus_idx
            hyper_params = self.cls_output[:,cur_idx:]
            hyper_params = hyper_params.reshape(N * 3, self.feature_num, 1, 1)
            hyper_params /= self.feature_num
            for i in range(0,div_img):
                for j in range(0,div_img):
                    org_img = org_img2[:,:,round(H2/div_img * i):round(H2/div_img * (i+1)), round(W2/div_img * j):round(W2/div_img * (j+1)) ]
                    N, C, H, W = org_img.shape
                    if self.hyper == 1:
                        org_img = org_img.reshape(1, N * 3, H, W)
                
                        if self.hyper_conv == 1:
                            org_img = F.conv2d(input=org_img, weight=transform_params, groups=N)

                        org_img = org_img.reshape(N,self.feature_num,H,W)
                        org_img = self.colorTransform(org_img, offset_param, index_image, color_map_control)
                    if self.mid_conv > 0:
                        org_img = self.mid_conv_module(org_img)
                    if self.last_hyper == 1:
                        org_img = org_img.reshape(1, N * self.feature_num, H, W)
                        org_img = F.conv2d(input=org_img, weight=hyper_params, groups=N)
                        org_img = org_img.reshape(N,3,H,W)
                    org_img2[:,:,round(H2/div_img * i):round(H2/div_img * (i+1)), round(W2/div_img * j):round(W2/div_img * (j+1)) ] = org_img


            if self.write_text == 0:
                return org_img2
            else:
                return org_img2, transform_params, offset_param, hyper_params
                    
                    
            


class DCPNet24_2(nn.Module):
    def __init__(self, config):
        super(DCPNet24_2, self).__init__()
        
        self.write_text = config.write_text

        self.control_point_num = config.control_point + 2
        self.feature_num = config.feature_num
        
        self.hyper = config.hyper
        self.xoffset = config.xoffset
        self.transform_num = config.transform_num
        self.conv_num = config.conv_num
        self.last_hyper = config.last_hyper

        self.residual = config.residual
        self.hyper_conv = config.hyper_conv

        self.local_residual = config.local_residual
        self.bias = config.bias
        self.quad = config.quad
        self.trans_param = config.trans_param
        if config.learnable_trans_param == 1:
            self.trans_param = nn.Parameter(torch.tensor([config.trans_param], dtype=torch.float32))

        self.leaky_relu = nn.LeakyReLU(0.1)
        param_num = (self.control_point_num * self.feature_num) * self.transform_num
        param_num1 = param_num
        param_num4 = 0
        if self.bias == 1:
            param_num += (self.feature_num * self.transform_num)
        if self.quad == 1 or self.quad == 4 or self.quad == 5:
            param_num += (self.feature_num * self.transform_num)
        if self.quad == 2 or self.quad == 3 or self.quad == 6 or self.quad == 7:
            param_num += (self.feature_num * self.transform_num) * 2
        if self.last_hyper == 1:
            param_num += (3 * self.feature_num)
            param_num4 += (3 * self.feature_num)
        param_xoffset = 0
        if self.xoffset == 1:
            param_xoffset = ((self.control_point_num - 2) * self.feature_num) * self.transform_num
            param_num += ((self.control_point_num - 2) * self.feature_num) * self.transform_num
        elif self.xoffset == 2:
            param_xoffset = ((self.control_point_num - 1) * self.feature_num) * self.transform_num
            param_num += ((self.control_point_num - 1) * self.feature_num) * self.transform_num

        if self.hyper == 0:
            if config.new_res == 0:
                self.classifier = resnet18_224(out_dim=param_num, res_size=config.res_size, res_num=config.res_num, fc_num=config.fc_num)
            elif config.new_res == 1:
                self.classifier = resnet18_224_2(out_dim=param_num, out_dim2=0, out_dim3=param_xoffset, out_dim4=param_num4, res_size=config.res_size, res_num=config.res_num,
                                               fc_num=config.fc_num, init_w=config.init_w, init_w2=config.init_w2, init_w_last=config.init_w_last, fc_node=config.fc_node, fc_node1=config.fc_node1, fc_node2=config.fc_node2)
            self.params = nn.Parameter(torch.randn(self.feature_num, 3, 1, 1))
        elif self.hyper == 1:
            if self.hyper_conv == 1:
                param_num += (3 * self.feature_num)
                param_num2 = (3 * self.feature_num)
            elif self.hyper_conv == 3:
                param_num += (3 * self.feature_num) * 9
                param_num2 = (3 * self.feature_num) * 9
            if config.new_res == 0:
                self.classifier = resnet18_224(out_dim=param_num, res_size=config.res_size, res_num=config.res_num, fc_num=config.fc_num)
            elif config.new_res == 1:
                self.classifier = resnet18_224_2(out_dim=param_num1, out_dim2=param_num2, out_dim3=param_xoffset, out_dim4=param_num4, res_size=config.res_size, res_num=config.res_num,
                                                 fc_num=config.fc_num, init_w=config.init_w, init_w2=config.init_w2, init_w_last=config.init_w_last, fc_node=config.fc_node, fc_node1=config.fc_node1, fc_node2=config.fc_node2)
                


        self.mid_conv = config.mid_conv
        conv_list = []
        for i in range(0, self.mid_conv):
            if config.mid_conv_mode == 'res':
                if config.mid_conv_size == 3:
                    conv_list.append(resBlock2(self.feature_num, self.feature_num, ksize=3, stride=1, pad=1, extra_conv=False, act='relu'))
                else:
                    conv_list.append(resBlock2(self.feature_num, self.feature_num, ksize=1, stride=1, pad=0, extra_conv=False, act='relu'))
            elif config.mid_conv_mode == 'res2':
                    conv_list.append(resBlock3(self.feature_num, self.feature_num, ksize=3, stride=1, pad=1, extra_conv=False, act='relu'))
            else:
                if config.mid_conv_size == 3:
                    ksize1 = 3
                    pad1 = 1
                elif config.mid_conv_size == 1:
                    ksize1 = 1
                    pad1 = 0
                if config.last_relu == 1:
                    act1 = 'relu'
                    bn1 = True
                else:
                    if i < self.mid_conv - 1:
                        act1 = 'relu'
                        bn1 = True
                    else:
                        act1 = 'none'
                        bn1 = False
                conv_list.append(convBlock2(self.feature_num, self.feature_num, ksize=ksize1, stride=1, pad=pad1, extra_conv=False, act=act1, bn=bn1))
        if self.mid_conv > 0:
            self.mid_conv_module = nn.Sequential(*conv_list)
        if config.xoffset == 0:
            self.colorTransform = colorTransform31(self.control_point_num, config.offset_param, config)
        elif config.xoffset == -1:
            self.colorTransform = colorTransform4(self.control_point_num, config.offset_param, config)
        elif config.xoffset == -2:
            self.colorTransform = colorTransform5(self.control_point_num, config.offset_param, config)
        elif config.xoffset == 1:
            self.colorTransform = colorTransform_xoffset(self.control_point_num, config.offset_param, config.offset_param2, config)
        elif config.xoffset == 2:
            self.colorTransform = colorTransform_xoffset_softmax(self.control_point_num, config.offset_param, config.offset_param2, config)
        if config.conv_mode == 3:
            self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=3, stride=1, padding=1).cuda(config.rank)
        elif config.conv_mode == 1:
            if config.last_conv_bias == 1:
                bias_flag = True
            else:
                bias_flag = False
            self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=1, stride=1, padding=0, bias=bias_flag).cuda(config.rank)
            if config.last_conv_init == 1:
                torch.nn.init.constant_(self.conv_out.weight.data, 1.0 / self.feature_num)
                if bias_flag:
                    torch.nn.init.constant_(self.conv_out.bias.data, 0)

        self.act = 'relu'
        self.pixelwise_multi = config.pixelwise_multi
        if self.pixelwise_multi == 1 or self.pixelwise_multi == 3 or self.pixelwise_multi == 4 or self.pixelwise_multi == 6:
            self.resize = T.Resize((config.local_size, config.local_size))
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

            self.res5 = convBlock2(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)  # s5

            self.res6 = convBlock2(input_feature=256, output_feature=256, ksize=3, stride=1, pad=1, act=self.act)  # s5

            # upsample and concat
            self.res7 = convBlock2(input_feature=384, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

            # upsample and concat
            self.res8 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

             # upsample and concat
            self.res9 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

            # upsample and concat
            self.res10 = convBlock2(input_feature=48, output_feature=self.feature_num, ksize=3, stride=1, pad=1, act=self.act)  # s1

            self.res11 = nn.Conv2d(self.feature_num, self.feature_num, kernel_size=3, padding=1)
        
        if self.pixelwise_multi == 2 or self.pixelwise_multi == 5:
            self.resize = T.Resize((config.local_size, config.local_size))
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

            self.res5 = convBlock2(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)  # s5

            self.res6 = convBlock2(input_feature=256, output_feature=256, ksize=3, stride=1, pad=1, act=self.act)  # s5

            # upsample and concat
            self.res7 = convBlock2(input_feature=384, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

            # upsample and concat
            self.res8 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

             # upsample and concat
            self.res9 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

            # upsample and concat
            self.res10 = convBlock2(input_feature=48, output_feature=self.feature_num * 2, ksize=3, stride=1, pad=1, act=self.act)  # s1

            self.res11 = nn.Conv2d(self.feature_num * 2, self.feature_num * 2, kernel_size=3, padding=1)

        self.div = config.div
        if self.local_residual == 1:
            if self.div == 4:
                self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
                self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
                self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

                self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
                # upsample and concat
                self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
                # upsample and concat
                self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
                self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True, act=self.act)
            elif self.div == 16:
                self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
                self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
                self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

                self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

                self.res5 = convBlock2(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)  # s5

                self.res6 = convBlock2(input_feature=256, output_feature=256, ksize=3, stride=1, pad=1, act=self.act)  # s5

                # upsample and concat
                self.res7 = convBlock2(input_feature=384, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

                # upsample and concat
                self.res8 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

                 # upsample and concat
                self.res9 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

                # upsample and concat
                self.res10 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1

                self.res11 = nn.Conv2d(16, 3, kernel_size=3, padding=1)

            elif self.div == 8:
                self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
                self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
                self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

                self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

                self.res5 = convBlock2(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

                # upsample and concat
                self.res6 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

                 # upsample and concat
                self.res7 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

                # upsample and concat
                self.res8 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1

                self.res9 = nn.Conv2d(16, 3, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.act_mode = config.act_mode

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, org_img, index_image, color_map_control):
        N, C, H, W = org_img.shape
        self.cls_output = self.classifier(org_img)
        if self.pixelwise_multi >= 1:
            y = self.resize(org_img)
            y1 = self.res1(y)
            y2 = self.res2(y1)
            y3 = self.res3(y2)
            y4 = self.res4(y3)
            y5 = self.res5(y4)
            y5 = self.res6(y5)
            y4 = self.res7(torch.cat((y4, F.upsample_bilinear(y5, scale_factor=2)), dim=1))
            y3 = self.res8(torch.cat((y3, F.upsample_bilinear(y4, scale_factor=2)), dim=1))
            y2 = self.res9(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
            y1 = self.res10(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
            y1 = self.res11(y1)
            m = T.Resize((H, W))
            y = m(y1)
        if self.hyper == 0:
            if self.act_mode == 'sigmoid':
                norm_params = self.sigmoid(self.trans_param * self.params)
                epsilon = 1e-10
                w_sum = torch.sum(torch.abs(norm_params), dim=1, keepdim=True)
                norm_params = norm_params / (w_sum + epsilon)
            elif self.act_mode == 'tanh':
                norm_params = self.tanh(self.params)
                epsilon = 1e-10
                w_norm = torch.norm(norm_params, dim=1, keepdim=True)
                norm_params = norm_params / (w_norm + epsilon)

            # 64 x 3 x 1 x 1
            img_f = F.conv2d(input=org_img, weight=norm_params)
            if self.act_mode == 'tanh':
                img_f = (img_f + 1) / 2
            img_f_t = self.colorTransform(img_f, self.cls_output, index_image, color_map_control)
            if self.act_mode == 'tanh':
                img_f_t = img_f_t * 2 -1
        elif self.hyper == 1:
            N, C, H, W = org_img.shape
            if self.hyper_conv == 1:
                cur_idx = 3 * self.feature_num
            elif self.hyper_conv == 3:
                cur_idx = 3 * self.feature_num * 9
            transform_params = self.cls_output[:,:cur_idx]
            if self.bias == 1:
                bias_params = self.cls_output[:,cur_idx:cur_idx + self.feature_num]
                cur_idx += self.feature_num
            if self.hyper_conv == 1:
                transform_params = transform_params.reshape(N * self.feature_num, 3)
            elif self.hyper_conv == 3:
                transform_params = transform_params.reshape(N * self.feature_num, 27)
            if self.act_mode == 'sigmoid':
                transform_params = self.sigmoid(self.trans_param * transform_params)
                epsilon = 1e-10
                t_sum = torch.sum(transform_params, dim=1, keepdim=True)
                transform_params = transform_params / (t_sum + epsilon)
            elif self.act_mode == 'tanh':
                transform_params = self.tanh(transform_params)
                epsilon = 1e-10
                w_norm = torch.sum(torch.abs(transform_params), dim=1, keepdim=True)
                transform_params = transform_params / (w_norm + epsilon)
            elif self.act_mode == 'minmax' or self.act_mode == 'trunc' or self.act_mode == 'minmax2':
                transform_params = transform_params

            if self.hyper_conv == 1:
                transform_params = transform_params.reshape(N * self.feature_num, 3, 1, 1)
            elif self.hyper_conv == 3:
                transform_params = transform_params.reshape(N * self.feature_num, 3, 3, 3)
            #transform_params = transform_params.permute(1,2,0,3,4)
            #org_img = org_img.permute(1,0,2,3)
            org_img = org_img.reshape(1, N * 3, H, W)
            if self.bias == 1:
                bias_params = bias_params.reshape(N * self.feature_num)
                if self.hyper_conv == 1:
                    img_f = F.conv2d(input=org_img, weight=transform_params, bias=bias_params, groups=N)
                elif self.hyper_conv == 3:
                    img_f = F.conv2d(input=org_img, weight=transform_params, bias=bias_params, groups=N, padding=1)
            else:
                if self.hyper_conv == 1:
                    org_img = F.conv2d(input=org_img, weight=transform_params, groups=N)
                elif self.hyper_conv == 3:
                    img_f = F.conv2d(input=org_img, weight=transform_params, groups=N, padding=1)

            org_img = org_img.reshape(N,self.feature_num,H,W)

            if self.quad == 1:
                quad_params = self.tanh(self.cls_output[:,cur_idx:cur_idx + self.feature_num])
                cur_idx += self.feature_num

                quad_params = quad_params.reshape(N, self.feature_num, 1, 1)
                img_f = img_f + quad_params * img_f * (1.0 - img_f)
            
            elif self.quad == 6 or self.quad == 7:
                gamma_params = self.cls_output[:,cur_idx:cur_idx + self.feature_num]
                cur_idx += self.feature_num
                beta_params = self.cls_output[:,cur_idx:cur_idx + self.feature_num]
                cur_idx += self.feature_num

                gamma_params = gamma_params.reshape(N, self.feature_num, 1, 1)
                beta_params = beta_params.reshape(N, self.feature_num, 1, 1)
                if self.quad == 6:
                    img_f = img_f * gamma_params + beta_params
                elif self.quad == 7:
                    img_f = img_f * gamma_params

            if self.pixelwise_multi == 1:
                img_f = img_f * y
            elif self.pixelwise_multi == 2:
                img_f = img_f * y[:,:self.feature_num,:,:] + y[:,self.feature_num:,:,:]
            elif self.pixelwise_multi == 3:
                y = self.tanh(y)
                img_f = img_f + y * img_f * ( 1.0 - img_f)
            if self.act_mode == 'minmax' or self.act_mode == 'minmax2':
                    # normalize
                min_val, _ = img_f.min(dim=2, keepdim=True)
                min_val, _ = min_val.min(dim=3, keepdim=True)

                max_val, _ = img_f.max(dim=2, keepdim=True)
                max_val, _ = max_val.max(dim=3, keepdim=True)
                img_f = (img_f - min_val) / (max_val - min_val)
            elif self.act_mode == 'trunc':
                img_f = torch.clamp(img_f, 0, 1)

            if self.act_mode == 'tanh':
                img_f = (img_f + 1.0) / 2
                img_f = torch.clamp(img_f, 0, 1)

            for i in range(0,self.transform_num):
                plus_idx = self.control_point_num * self.feature_num
                if self.xoffset == 1:
                    plus_idx += ((self.control_point_num - 2) * self.feature_num)
                elif self.xoffset == 2:
                    plus_idx += ((self.control_point_num - 1) * self.feature_num)
                offset_param = self.cls_output[:,cur_idx:cur_idx + plus_idx]
                cur_idx += plus_idx
                org_img = self.colorTransform(org_img, offset_param, index_image, color_map_control)
                if self.act_mode == 'minmax2':
                    img_f_t = min_val + (max_val - min_val) * img_f_t
                if i < self.transform_num - 1:
                    # normalize
                    min_val, _ = img_f_t.min(dim=2, keepdim=True)
                    min_val, _ = min_val.min(dim=3, keepdim=True)

                    max_val, _ = img_f_t.max(dim=2, keepdim=True)
                    max_val, _ = max_val.max(dim=3, keepdim=True)
                    img_f_t = (img_f_t - min_val) / (max_val - min_val)
                    if self.act_mode == 'tanh':
                        img_f_t = img_f_t * 2.0 - 1

        if self.pixelwise_multi == 4:
            img_f_t = img_f_t * y
        elif self.pixelwise_multi == 5:
            img_f_t = img_f_t * y[:,:self.feature_num,:,:] + y[:,self.feature_num:,:,:]
        elif self.pixelwise_multi == 6:
            y = self.tanh(y)
            img_f_t = img_f_t + y * img_f_t * ( 1.0 - img_f_t)

        if self.mid_conv > 0:
            org_img = self.mid_conv_module(org_img)
        if self.last_hyper == 1:
            hyper_params = self.cls_output[:,cur_idx:]
            hyper_params = hyper_params.reshape(N * 3, self.feature_num, 1, 1)
            #hyper_params = self.tanh(hyper_params)
            hyper_params /= self.feature_num

            org_img = org_img.reshape(1, N * self.feature_num, H, W)
            org_img = F.conv2d(input=org_img, weight=hyper_params, groups=N)
            org_img = org_img.reshape(N,3,H,W)
        else:
            out_img = self.conv_out(img_f_t)

        if self.quad == 2 or self.quad == 3:
            gamma_params = self.cls_output[:,cur_idx:cur_idx + self.feature_num]
            cur_idx += self.feature_num
            beta_params = self.cls_output[:,cur_idx:cur_idx + self.feature_num]
            cur_idx += self.feature_num

            gamma_params = gamma_params.reshape(N, self.feature_num, 1, 1)
            beta_params = beta_params.reshape(N, self.feature_num, 1, 1)
            img_f_t = img_f_t * gamma_params + beta_params
            if self.quad == 3:
                img_f_t = self.relu(img_f_t)
        elif self.quad == 4:
            quad_params = self.tanh(self.cls_output[:,cur_idx:cur_idx + self.feature_num])
            cur_idx += self.feature_num
            quad_params = quad_params.reshape(N, self.feature_num, 1, 1)
            img_f_t = img_f_t + quad_params * img_f_t * (1.0 - img_f_t)
        elif self.quad == 5:
            gamma_params = self.sigmoid(self.cls_output[:,cur_idx:cur_idx + self.feature_num])
            cur_idx += self.feature_num 
            gamma_params = gamma_params.reshape(N, self.feature_num, 1, 1)
            img_f_t = img_f_t * gamma_params


        #img_f = self.conv_emb(org_img)
        if self.residual == 1:
            org_img = org_img.reshape(N, 3, H, W)
            out_img = out_img + org_img

        if self.local_residual == 1:
            if self.div == 4:
                y1 = self.res1(out_img)
                y2 = self.res2(y1)
                y3 = self.res3(y2)
                y3 = self.res4(y3)
                y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
                y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
                y1 = self.res7(y1)
            elif self.div == 16:
                y1 = self.res1(out_img)
                y2 = self.res2(y1)
                y3 = self.res3(y2)
                y4 = self.res4(y3)
                y5 = self.res5(y4)
                y5 = self.res6(y5)
                y4 = self.res7(torch.cat((y4, F.upsample_bilinear(y5, scale_factor=2)), dim=1))
                y3 = self.res8(torch.cat((y3, F.upsample_bilinear(y4, scale_factor=2)), dim=1))
                y2 = self.res9(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
                y1 = self.res10(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
                y1 = self.res11(y1)
            elif self.div == 8:
                y1 = self.res1(out_img)
                y2 = self.res2(y1)
                y3 = self.res3(y2)
                y4 = self.res4(y3)
                y4 = self.res5(y4)
                y3 = self.res6(torch.cat((y3, F.upsample_bilinear(y4, scale_factor=2)), dim=1))
                y2 = self.res7(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
                y1 = self.res8(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
                y1 = self.res9(y1)

            out_img = y1 + out_img
        if self.write_text == 0:
            return org_img
        else:
            return org_img, transform_params, offset_param, hyper_params

class DCPNet240(nn.Module):
    def __init__(self, config):
        super(DCPNet240, self).__init__()

        self.control_point_num = config.control_point + 2
        self.feature_num = config.feature_num
        
        self.hyper = config.hyper
        self.conv_num = config.conv_num
        self.last_hyper = config.last_hyper

        self.hyper_conv = config.hyper_conv

        self.trans_param = config.trans_param

        param_num = self.control_point_num * self.feature_num
        param_num1 = param_num
        param_num4 = 0

        if self.last_hyper == 1:
            param_num += (3 * self.feature_num)
            param_num4 += (3 * self.feature_num)

        if self.hyper == 0:
            if config.new_res == 0:
                self.classifier = resnet18_224(out_dim=param_num, res_size=config.res_size, res_num=config.res_num, fc_num=config.fc_num)
            elif config.new_res == 1:
                self.classifier = resnet18_224_2(out_dim=param_num, out_dim2=0, out_dim3=0, out_dim4=param_num4, res_size=config.res_size, res_num=config.res_num,
                                               fc_num=config.fc_num, init_w=config.init_w, init_w2=config.init_w2, init_w_last=config.init_w_last, fc_node=config.fc_node, fc_node1=config.fc_node1, fc_node2=config.fc_node2)
            self.params = nn.Parameter(torch.randn(self.feature_num, 3, 1, 1))
        elif self.hyper == 1:
            if self.hyper_conv == 1:
                param_num += (3 * self.feature_num)
                param_num2 = (3 * self.feature_num)
            elif self.hyper_conv == 3:
                param_num += (3 * self.feature_num) * 9
                param_num2 = (3 * self.feature_num) * 9
            self.classifier = resnet18_224_2(out_dim=param_num1, out_dim2=param_num2, out_dim3=0, out_dim4=param_num4, res_size=config.res_size, res_num=config.res_num, fc_num=config.fc_num, init_w=config.init_w, init_w2=config.init_w2, init_w_last=config.init_w_last, fc_node=config.fc_node, fc_node1=config.fc_node1, fc_node2=config.fc_node2)


        self.mid_conv = config.mid_conv
        conv_list = []
        for i in range(0, self.mid_conv):
            if config.mid_conv_mode == 'res':
                if config.mid_conv_size == 3:
                    conv_list.append(resBlock2(self.feature_num, self.feature_num, ksize=3, stride=1, pad=1, extra_conv=False, act='relu'))
                else:
                    conv_list.append(resBlock2(self.feature_num, self.feature_num, ksize=1, stride=1, pad=0, extra_conv=False, act='relu'))
            elif config.mid_conv_mode == 'res2':
                    conv_list.append(resBlock3(self.feature_num, self.feature_num, ksize=3, stride=1, pad=1, extra_conv=False, act='relu'))
            else:
                if config.mid_conv_size == 3:
                    ksize1 = 3
                    pad1 = 1
                elif config.mid_conv_size == 1:
                    ksize1 = 1
                    pad1 = 0
                if config.last_relu == 1:
                    act1 = 'relu'
                    bn1 = True
                else:
                    if i < self.mid_conv - 1:
                        act1 = 'relu'
                        bn1 = True
                    else:
                        act1 = 'none'
                        bn1 = False
                conv_list.append(convBlock2(self.feature_num, self.feature_num, ksize=ksize1, stride=1, pad=pad1, extra_conv=False, act=act1, bn=bn1))
        if self.mid_conv > 0:
            self.mid_conv_module = nn.Sequential(*conv_list)
        if config.xoffset == 0:
            self.colorTransform = colorTransform3(self.control_point_num, config.offset_param, config)
        elif config.xoffset == -100:
            self.colorTransform = colorTransform_no(self.control_point_num, config.offset_param, config)

        if config.conv_mode == 3:
            self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=3, stride=1, padding=1).cuda(config.rank)
        elif config.conv_mode == 1:
            if config.last_conv_bias == 1:
                bias_flag = True
            else:
                bias_flag = False
            if config.last_hyper == 0:
                self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=1, stride=1, padding=0, bias=bias_flag).cuda(config.rank)

        self.sigmoid = nn.Sigmoid()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, org_img, index_image, color_map_control):
        N, C, H, W = org_img.shape
        self.cls_output = self.classifier(org_img)
        if self.hyper == 0:
            norm_params = self.sigmoid(self.trans_param * self.params)
            epsilon = 1e-10
            w_sum = torch.sum(torch.abs(norm_params), dim=1, keepdim=True)
            norm_params = norm_params / (w_sum + epsilon)
            # 64 x 3 x 1 x 1
            img_f = F.conv2d(input=org_img, weight=norm_params)
        elif self.hyper == 1:
            N, C, H, W = org_img.shape
            if self.hyper_conv == 1:
                cur_idx = 3 * self.feature_num
            elif self.hyper_conv == 3:
                cur_idx = 3 * self.feature_num * 9
            transform_params = self.cls_output[:,:cur_idx]
            if self.hyper_conv == 1:
                transform_params = transform_params.reshape(N * self.feature_num, 3)
            elif self.hyper_conv == 3:
                transform_params = transform_params.reshape(N * self.feature_num, 27)

            transform_params = self.sigmoid(self.trans_param * transform_params)
            epsilon = 1e-10
            t_sum = torch.sum(transform_params, dim=1, keepdim=True)
            transform_params = transform_params / (t_sum + epsilon)


            if self.hyper_conv == 1:
                transform_params = transform_params.reshape(N * self.feature_num, 3, 1, 1)
            elif self.hyper_conv == 3:
                transform_params = transform_params.reshape(N * self.feature_num, 3, 3, 3)
            org_img = org_img.reshape(1, N * 3, H, W)
            if self.hyper_conv == 1:
                img_f = F.conv2d(input=org_img, weight=transform_params, groups=N)
            elif self.hyper_conv == 3:
                img_f = F.conv2d(input=org_img, weight=transform_params, groups=N, padding=1)

            img_f = img_f.reshape(N,self.feature_num,H,W)

            
            plus_idx = self.control_point_num * self.feature_num
            offset_param = self.cls_output[:,cur_idx:cur_idx + plus_idx]
            cur_idx += plus_idx
            img_f_t = self.colorTransform(img_f, offset_param, index_image, color_map_control)
        
        if self.mid_conv > 0:
            img_f_t = self.mid_conv_module(img_f_t)
        if self.last_hyper == 1:
            hyper_params = self.cls_output[:,cur_idx:]
            hyper_params = hyper_params.reshape(N * 3, self.feature_num, 1, 1)
            hyper_params /= self.feature_num

            img_f_t = img_f_t.reshape(1, N * self.feature_num, H, W)
            out_img = F.conv2d(input=img_f_t, weight=hyper_params, groups=N)
            out_img = out_img.reshape(N,3,H,W)
        else:
            out_img = self.conv_out(img_f_t)

        return out_img

class DCPNet25(nn.Module):
    def __init__(self, config):
        super(DCPNet25, self).__init__()

        self.control_point_num = config.control_point + 2
        self.feature_num = config.feature_num
        
        self.hyper = config.hyper
        self.xoffset = config.xoffset

        self.num_weight = config.num_weight
        self.hyper2 = config.hyper2

        if self.hyper2 == 0:
            self.offset_params = nn.Parameter(torch.randn(1,self.control_point_num * self.feature_num * self.num_weight) * 0.04) 

        self.param_num = self.control_point_num * self.feature_num * config.num_weight
        self.res_guide = config.res_guide
        if config.res_guide == 0:
            self.add_num = 0
        else:
            self.add_num = 128
        if self.xoffset == 1:
            self.param_num += (self.control_point_num - 2) * self.feature_num
        if self.hyper == 0:    
            self.classifier = resnet18_224(out_dim=self.param_num+self.add_num, res_size=config.res_size, res_num=config.res_num)
            self.params = nn.Parameter(torch.randn(self.feature_num, 3, 1, 1))
        elif self.hyper == 1:
            self.param_num += (3 * self.feature_num)
            self.classifier = resnet18_224(out_dim=self.param_num+self.add_num, res_size=config.res_size, res_num=config.res_num)

        
        if self.num_weight > 1:
            self.colorTransform = colorTransform_multi(self.control_point_num, config.offset_param, self.num_weight, config)
        else:    
            if config.xoffset == 0:
                self.colorTransform = colorTransform3(self.control_point_num, config.offset_param, config)
            else:
                self.colorTransform = colorTransform_xoffset(self.control_point_num, config.offset_param, config)
        if config.conv_mode == 3:
            self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=3, stride=1, padding=1)
        elif config.conv_mode == 1:
            self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=1, stride=1, padding=0)
        
        
        self.act = 'leaky'
        if self.num_weight > 1:
            self.resize = T.Resize((config.local_size, config.local_size))
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

            
            self.res5 = convBlock2(input_feature=128 + self.add_num, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

            # upsample and concat
            self.res6 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

             # upsample and concat
            self.res7 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

            # upsample and concat
            self.res8 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1

            self.res9 = nn.Conv2d(16, self.num_weight, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, org_img, index_image, color_map_control):
        N, C, H, W = org_img.shape

        self.cls_output = self.classifier(org_img)
        
        if self.num_weight > 1:
            
            y = self.resize(org_img)
            y1 = self.res1(y)
            y2 = self.res2(y1)
            y3 = self.res3(y2)
            y4 = self.res4(y3)
            if self.res_guide == 1:
                _,_,nh,nw = y4.shape
                add_feat = self.cls_output[:,self.param_num:]
                add_feat = add_feat.reshape(N,self.add_num,1,1)
                add_feat = add_feat.repeat(1,1,nh,nw)
                y4 = torch.cat((y4,add_feat), dim=1)
            y4 = self.res5(y4)
            y3 = self.res6(torch.cat((y3, F.upsample_bilinear(y4, scale_factor=2)), dim=1))
            y2 = self.res7(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
            y1 = self.res8(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
            y1 = self.res9(y1)
            m = T.Resize((H, W))
            y = m(y1)
            y = self.sigmoid(y)
            y_sum = torch.sum(y, dim=1, keepdim=True)
            y = y / y_sum
            

        

        if self.hyper == 0:
            norm_params = self.sigmoid(self.params)
            epsilon = 1e-10
            w_sum = torch.sum(norm_params, dim=1, keepdim=True)
            norm_params = norm_params / (w_sum + epsilon)
            # 64 x 3 x 1 x 1
            img_f = F.conv2d(input=org_img, weight=norm_params)
            if self.num_weight == 1:
                img_f_t = self.colorTransform(img_f, self.cls_output[:,:self.param_num], index_image, color_map_control)
            else:
                if self.hyper2 == 0:
                    self.offset_params_repeat = self.offset_params.repeat(N,1)
                    img_f_t = self.colorTransform(img_f, self.offset_params_repeat, index_image, color_map_control, y)
                else:
                    img_f_t = self.colorTransform(img_f, self.cls_output[:,:self.param_num], index_image, color_map_control, y)
        elif self.hyper == 1:
            transform_params = self.cls_output[:,:3 * self.feature_num]
            offset_param = self.cls_output[:,3 * self.feature_num:self.param_num]
            
            transform_params = transform_params.reshape(N * self.feature_num, 3)
            transform_params = self.sigmoid(transform_params)
            epsilon = 1e-10
            t_sum = torch.sum(transform_params, dim=1, keepdim=True)
            transform_params = transform_params / (t_sum + epsilon)
            transform_params = transform_params.reshape(N * self.feature_num, 3,1,1)
            #transform_params = transform_params.permute(1,2,0,3,4)
            #org_img = org_img.permute(1,0,2,3)
            org_img = org_img.reshape(1, N * 3, H, W)
            
            img_f = F.conv2d(input=org_img, weight=transform_params, groups=N)
            img_f = img_f.reshape(N,self.feature_num,H,W)

            if self.num_weight == 1:
                img_f_t = self.colorTransform(img_f, offset_param, index_image, color_map_control)
            else:
                if self.hyper2 == 0:
                    self.offset_params_repeat = self.offset_params.repeat(N,1)
                    img_f_t = self.colorTransform(img_f, self.offset_params_repeat, index_image, color_map_control, y)
                else:
                    img_f_t = self.colorTransform(img_f, offset_param, index_image, color_map_control, y)

        
        #self.conv_emb = F.conv2d(3, self.feature_num, weight=norm_params, kernel_size=1, stride=1, padding=0, bias=False)
        #self.temp_weight =
        #conv_emb = nn.Conv2d(3, self.feature_num, weight= , kernel_size=1, stride=1, padding=0, bias=False)
        out_img = self.conv_out(img_f_t)

        #img_f = self.conv_emb(org_img)

        return out_img


class DCPNet26(nn.Module):
    def __init__(self, config):
        super(DCPNet26, self).__init__()

        self.control_point_num = config.control_point + 2
        self.feature_num = config.feature_num

        self.hyper = config.hyper
        self.xoffset = config.xoffset

        self.conv_num = config.conv_num
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.act = 'leaky'
        self.backbone = config.backbone
        param_num = self.control_point_num * self.feature_num
        if self.xoffset == 1:
            param_num += (self.control_point_num - 2) * self.feature_num
        if self.hyper == 0:
            self.classifier = resnet18_224(out_dim=param_num, res_size=config.res_size, res_num=config.res_num)
            self.params = nn.Parameter(torch.randn(self.feature_num, 3, 1, 1))
        elif self.hyper == 1:
            param_num += (3 * self.feature_num)
            if self.conv_num > 1:
                param_num += ((self.feature_num * self.feature_num) * (self.conv_num - 1))
            if self.backbone == 'res':
                self.classifier = resnet18_224(out_dim=param_num, res_size=config.res_size, res_num=config.res_num)
            elif self.backbone == 'vit':
                lists = []
                lists.append(nn.Upsample(size=(config.patch_size, config.patch_size), mode='bilinear'))
                lists.append(convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act))
                lists.append(convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act))
                lists.append(convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act))
                lists.append(convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act))
                lists.append(ViT2(image_size=config.patch_size, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512, pool='cls', dim_head=64, dropout=0.1))
                self.color_conv_global = convBlock2(input_feature=128, output_feature=param_num, ksize=1, stride=1, pad=0, extra_conv=True, act=self.act)
                #lists.append(convBlock2(input_feature=128, output_feature=param_num, ksize=1, stride=1, pad=0, extra_conv=True, act=self.act))
                self.classifier = nn.Sequential(*lists)


        if config.xoffset == 0:
            self.colorTransform = colorTransform3(self.control_point_num, config.offset_param, config)
        else:
            self.colorTransform = colorTransform_xoffset(self.control_point_num, config.offset_param, config)
        if config.conv_mode == 3:
            self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=3, stride=1, padding=1)
        elif config.conv_mode == 1:
            self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.act_mode = config.act_mode

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, org_img, index_image, color_map_control):
        if self.backbone == 'res':
            self.cls_output = self.classifier(org_img)
        elif self.backbone == 'vit':
            x5 = self.classifier(org_img)
            x5 = torch.transpose(x5, 1, 2)
            x5_global = x5[:, :, 0]
            x5_global = x5_global.unsqueeze(2).unsqueeze(3)
            self.cls_output = self.color_conv_global(x5_global)
        if self.hyper == 0:
            if self.act_mode == 'sigmoid':
                norm_params = self.sigmoid(self.params)
                epsilon = 1e-10
                w_sum = torch.sum(torch.abs(norm_params), dim=1, keepdim=True)
                norm_params = norm_params / (w_sum + epsilon)
            elif self.act_mode == 'tanh':
                norm_params = self.tanh(self.params)
                epsilon = 1e-10
                w_norm = torch.norm(norm_params, dim=1, keepdim=True)
                norm_params = norm_params / (w_norm + epsilon)

            # 64 x 3 x 1 x 1
            img_f = F.conv2d(input=org_img, weight=norm_params)
            if self.act_mode == 'tanh':
                img_f = (img_f + 1) / 2
            img_f_t = self.colorTransform(img_f, self.cls_output, index_image, color_map_control)
            if self.act_mode == 'tanh':
                img_f_t = img_f_t * 2 - 1
        elif self.hyper == 1:
            N, C, H, W = org_img.shape
            cur_idx = 3 * self.feature_num
            if self.conv_num > 1:
                cur_idx += ((self.feature_num * self.feature_num) * (self.conv_num - 1))
            transform_params = self.cls_output[:, :cur_idx]
            offset_param = self.cls_output[:, cur_idx:]

            transform_params1 = transform_params[:, :self.feature_num * 3]
            transform_params1 = transform_params1.reshape(N * self.feature_num, 3)
            if self.act_mode == 'sigmoid':
                transform_params1 = self.sigmoid(transform_params1)
                epsilon = 1e-10
                t_sum = torch.sum(transform_params1, dim=1, keepdim=True)
                transform_params1 = transform_params1 / (t_sum + epsilon)
            elif self.act_mode == 'tanh':
                transform_params1 = self.tanh(transform_params1)
                epsilon = 1e-10
                w_norm = torch.sum(torch.abs(transform_params1), dim=1, keepdim=True)
                transform_params1 = transform_params1 / (w_norm + epsilon)
            elif self.act_mode == 'none':
                transform_params1 = transform_params1

            transform_params1 = transform_params1.reshape(N * self.feature_num, 3, 1, 1)
            # transform_params1 = transform_params1.permute(1,2,0,3,4)
            # org_img = org_img.permute(1,0,2,3)
            org_img = org_img.reshape(1, N * 3, H, W)
            img_f = F.conv2d(input=org_img, weight=transform_params1, groups=N)
            if self.conv_num == 1:
                if self.act_mode == 'none':
                    img_f = self.sigmoid(img_f)
            elif self.conv_num > 1:
                img_f = self.leaky_relu(img_f)
                cur_idx = 3 * self.feature_num
                for c in range(0, self.conv_num - 1):
                    add_transform_params = transform_params[:, cur_idx:cur_idx + self.feature_num ** 2]
                    cur_idx += self.feature_num ** 2
                    add_transform_params = add_transform_params.reshape(N * self.feature_num, self.feature_num, 1, 1)
                    img_f = F.conv2d(input=img_f, weight=add_transform_params, groups=N)
                img_f = self.sigmoid(img_f)

            img_f = img_f.reshape(N, self.feature_num, H, W)
            if self.act_mode == 'tanh':
                img_f = (img_f + 1.0) / 2
                img_f = torch.clamp(img_f, 0, 1)
            img_f_t = self.colorTransform(img_f, offset_param, index_image, color_map_control)
            if self.act_mode == 'tanh':
                img_f_t = img_f_t * 2.0 - 1

        # self.conv_emb = F.conv2d(3, self.feature_num, weight=norm_params, kernel_size=1, stride=1, padding=0, bias=False)
        # self.temp_weight =
        # conv_emb = nn.Conv2d(3, self.feature_num, weight= , kernel_size=1, stride=1, padding=0, bias=False)
        out_img = self.conv_out(img_f_t)

        # img_f = self.conv_emb(org_img)

        return out_img


class DCPNet27(nn.Module):
    def __init__(self, config):
        super(DCPNet27, self).__init__()

        self.control_point_num = config.control_point + 2
        self.feature_num = config.feature_num
        
        self.hyper = config.hyper
        self.xoffset = config.xoffset
        self.transform_num = config.transform_num
        self.conv_num = config.conv_num
        self.last_hyper = config.last_hyper

        self.residual = config.residual
        self.hyper_conv = config.hyper_conv

        self.local_residual = config.local_residual
        self.bias = config.bias
        self.quad = config.quad
        self.trans_param = config.trans_param
        if config.learnable_trans_param == 1:
            self.trans_param = nn.Parameter(torch.tensor([config.trans_param], dtype=torch.float32))

        self.softmax = config.softmax
        self.num_weight = config.num_weight

        self.leaky_relu = nn.LeakyReLU(0.1)
        param_num = (self.control_point_num * self.feature_num) * self.transform_num * config.num_weight
        param_num1 = param_num
        param_num4 = 0
        if self.bias == 1:
            param_num += (self.feature_num * self.transform_num)
        if self.quad == 1 or self.quad == 4 or self.quad == 5:
            param_num += (self.feature_num * self.transform_num)
        if self.quad == 2 or self.quad == 3 or self.quad == 6 or self.quad == 7:
            param_num += (self.feature_num * self.transform_num) * 2
        if self.last_hyper == 1:
            param_num += (3 * self.feature_num)
            param_num4 += (3 * self.feature_num)
        param_xoffset = 0
        if self.xoffset == 1:
            param_xoffset = ((self.control_point_num - 2) * self.feature_num) * self.transform_num
            param_num += ((self.control_point_num - 2) * self.feature_num) * self.transform_num
        elif self.xoffset == 2:
            param_xoffset = ((self.control_point_num - 1) * self.feature_num) * self.transform_num
            param_num += ((self.control_point_num - 1) * self.feature_num) * self.transform_num

        if self.hyper == 0:
            if config.new_res == 0:
                self.classifier = resnet18_224(out_dim=param_num, res_size=config.res_size, res_num=config.res_num, fc_num=config.fc_num)
            elif config.new_res == 1:
                self.classifier = resnet18_224_2(out_dim=param_num, out_dim2=0, out_dim3=param_xoffset, out_dim4=param_num4, res_size=config.res_size, res_num=config.res_num,
                                               fc_num=config.fc_num, init_w=config.init_w, init_w2=config.init_w2)
            self.params = nn.Parameter(torch.randn(self.feature_num, 3, 1, 1))
        elif self.hyper == 1:
            if self.hyper_conv == 1:
                param_num += (3 * self.feature_num)
                param_num2 = (3 * self.feature_num)
            elif self.hyper_conv == 3:
                param_num += (3 * self.feature_num) * 9
                param_num2 = (3 * self.feature_num) * 9
            if config.new_res == 0:
                self.classifier = resnet18_224(out_dim=param_num, res_size=config.res_size, res_num=config.res_num, fc_num=config.fc_num)
            elif config.new_res == 1:
                self.classifier = resnet18_224_2(out_dim=param_num1, out_dim2=param_num2, out_dim3=param_xoffset, out_dim4=param_num4, res_size=config.res_size, res_num=config.res_num,
                                                 fc_num=config.fc_num, init_w=config.init_w, init_w2=config.init_w2)


        self.mid_conv = config.mid_conv
        conv_list = []
        for i in range(0, self.mid_conv):
            if config.mid_conv_mode == 'res':
                if config.mid_conv_size == 3:
                    conv_list.append(resBlock2(self.feature_num, self.feature_num, ksize=3, stride=1, pad=1, extra_conv=False, act='relu'))
                else:
                    conv_list.append(resBlock2(self.feature_num, self.feature_num, ksize=1, stride=1, pad=0, extra_conv=False, act='relu'))
            elif config.mid_conv_mode == 'res2':
                    conv_list.append(resBlock3(self.feature_num, self.feature_num, ksize=3, stride=1, pad=1, extra_conv=False, act='relu'))
            else:
                if config.mid_conv_size == 3:
                    conv_list.append(convBlock2(self.feature_num, self.feature_num, ksize=3, stride=1, pad=1, extra_conv=False, act='relu'))
                else:
                    conv_list.append(convBlock2(self.feature_num, self.feature_num, ksize=1, stride=1, pad=0, extra_conv=False, act='relu'))
        if self.mid_conv > 0:
            self.mid_conv_module = nn.Sequential(*conv_list)
        if self.num_weight > 1:
            self.colorTransform = colorTransform_multi(self.control_point_num, config.offset_param, self.num_weight, config)
        else:
            if config.xoffset == 0:
                self.colorTransform = colorTransform3(self.control_point_num, config.offset_param, config)
            elif config.xoffset == -1:
                self.colorTransform = colorTransform4(self.control_point_num, config.offset_param, config)
            elif config.xoffset == -2:
                self.colorTransform = colorTransform5(self.control_point_num, config.offset_param, config)
            elif config.xoffset == 1:
                self.colorTransform = colorTransform_xoffset(self.control_point_num, config.offset_param, config)
            elif config.xoffset == 2:
                self.colorTransform = colorTransform_xoffset_softmax(self.control_point_num, config.offset_param, config.offset_param2, config)
        if config.conv_mode == 3:
            self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=3, stride=1, padding=1).cuda(config.rank)
        elif config.conv_mode == 1:
            self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=1, stride=1, padding=0).cuda(config.rank)
        
        
        self.act = 'leaky'
        if self.num_weight > 1:
            self.resize = T.Resize((config.local_size, config.local_size))
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

            self.res5 = convBlock2(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)  # s5

            self.res6 = convBlock2(input_feature=256, output_feature=256, ksize=3, stride=1, pad=1, act=self.act)  # s5

            # upsample and concat
            self.res7 = convBlock2(input_feature=384, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

            # upsample and concat
            self.res8 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

             # upsample and concat
            self.res9 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

            # upsample and concat
            self.res10 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1

            self.res11 = nn.Conv2d(16, self.num_weight, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, org_img, index_image, color_map_control):
        N, C, H, W = org_img.shape
        if self.num_weight > 1:
            y = self.resize(org_img)
            y1 = self.res1(y)
            y2 = self.res2(y1)
            y3 = self.res3(y2)
            y4 = self.res4(y3)
            y5 = self.res5(y4)
            y5 = self.res6(y5)
            y4 = self.res7(torch.cat((y4, F.upsample_bilinear(y5, scale_factor=2)), dim=1))
            y3 = self.res8(torch.cat((y3, F.upsample_bilinear(y4, scale_factor=2)), dim=1))
            y2 = self.res9(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
            y1 = self.res10(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
            y1 = self.res11(y1)

            m = T.Resize((H, W))
            y = m(y1)
            if self.softmax == 0:
                y = self.sigmoid(y)
                y_sum = torch.sum(y, dim=1, keepdim=True)
                y = y / y_sum
            else:
                m = nn.Softmax(dim=1)
                y = m(y)
            

        N, C, H, W = org_img.shape
        self.cls_output = self.classifier(org_img)

        if self.hyper == 0:
            if self.act_mode == 'sigmoid':
                norm_params = self.sigmoid(self.trans_param * self.params)
                epsilon = 1e-10
                w_sum = torch.sum(torch.abs(norm_params), dim=1, keepdim=True)
                norm_params = norm_params / (w_sum + epsilon)
            elif self.act_mode == 'tanh':
                norm_params = self.tanh(self.params)
                epsilon = 1e-10
                w_norm = torch.norm(norm_params, dim=1, keepdim=True)
                norm_params = norm_params / (w_norm + epsilon)

            # 64 x 3 x 1 x 1
            img_f = F.conv2d(input=org_img, weight=norm_params)
            if self.act_mode == 'tanh':
                img_f = (img_f + 1) / 2
            if self.num_weight > 1:
                img_f_t = self.colorTransform(img_f, self.cls_output, index_image, color_map_control, y)
            else:
                img_f_t = self.colorTransform(img_f, self.cls_output, index_image, color_map_control)
            if self.act_mode == 'tanh':
                img_f_t = img_f_t * 2 -1
        elif self.hyper == 1:
            N, C, H, W = org_img.shape
            if self.hyper_conv == 1:
                cur_idx = 3 * self.feature_num
            elif self.hyper_conv == 3:
                cur_idx = 3 * self.feature_num * 9
            transform_params = self.cls_output[:,:cur_idx]
            if self.bias == 1:
                bias_params = self.cls_output[:,cur_idx:cur_idx + self.feature_num]
                cur_idx += self.feature_num
            if self.hyper_conv == 1:
                transform_params = transform_params.reshape(N * self.feature_num, 3)
            elif self.hyper_conv == 3:
                transform_params = transform_params.reshape(N * self.feature_num, 27)

            transform_params = self.sigmoid(self.trans_param * transform_params)
            epsilon = 1e-10
            t_sum = torch.sum(transform_params, dim=1, keepdim=True)
            transform_params = transform_params / (t_sum + epsilon)


            if self.hyper_conv == 1:
                transform_params = transform_params.reshape(N * self.feature_num, 3, 1, 1)
            elif self.hyper_conv == 3:
                transform_params = transform_params.reshape(N * self.feature_num, 3, 3, 3)
            #transform_params = transform_params.permute(1,2,0,3,4)
            #org_img = org_img.permute(1,0,2,3)
            org_img = org_img.reshape(1, N * 3, H, W)
            if self.bias == 1:
                bias_params = bias_params.reshape(N * self.feature_num)
                if self.hyper_conv == 1:
                    img_f = F.conv2d(input=org_img, weight=transform_params, bias=bias_params, groups=N)
                elif self.hyper_conv == 3:
                    img_f = F.conv2d(input=org_img, weight=transform_params, bias=bias_params, groups=N, padding=1)
            else:
                if self.hyper_conv == 1:
                    img_f = F.conv2d(input=org_img, weight=transform_params, groups=N)
                elif self.hyper_conv == 3:
                    img_f = F.conv2d(input=org_img, weight=transform_params, groups=N, padding=1)

            img_f = img_f.reshape(N,self.feature_num,H,W)

            if self.quad == 1:
                quad_params = self.tanh(self.cls_output[:,cur_idx:cur_idx + self.feature_num])
                cur_idx += self.feature_num

                quad_params = quad_params.reshape(N, self.feature_num, 1, 1)
                img_f = img_f + quad_params * img_f * (1.0 - img_f)
            
            elif self.quad == 6 or self.quad == 7:
                gamma_params = self.cls_output[:,cur_idx:cur_idx + self.feature_num]
                cur_idx += self.feature_num
                beta_params = self.cls_output[:,cur_idx:cur_idx + self.feature_num]
                cur_idx += self.feature_num

                gamma_params = gamma_params.reshape(N, self.feature_num, 1, 1)
                beta_params = beta_params.reshape(N, self.feature_num, 1, 1)
                if self.quad == 6:
                    img_f = img_f * gamma_params + beta_params
                elif self.quad == 7:
                    img_f = img_f * gamma_params



            for i in range(0,self.transform_num):
                plus_idx = self.control_point_num * self.feature_num * self.num_weight
                if self.xoffset == 1:
                    plus_idx += ((self.control_point_num - 2) * self.feature_num)
                elif self.xoffset == 2:
                    plus_idx += ((self.control_point_num - 1) * self.feature_num)
                offset_param = self.cls_output[:,cur_idx:cur_idx + plus_idx]
                cur_idx += plus_idx
                if self.num_weight > 1:
                    img_f_t = self.colorTransform(img_f, offset_param, index_image, color_map_control, y)
                else:
                    img_f_t = self.colorTransform(img_f, offset_param, index_image, color_map_control)

                if i < self.transform_num - 1:
                    # normalize
                    min_val, _ = img_f_t.min(dim=2, keepdim=True)
                    min_val, _ = min_val.min(dim=3, keepdim=True)

                    max_val, _ = img_f_t.max(dim=2, keepdim=True)
                    max_val, _ = max_val.max(dim=3, keepdim=True)
                    img_f_t = (img_f_t - min_val) / (max_val - min_val)
                    if self.act_mode == 'tanh':
                        img_f_t = img_f_t * 2.0 - 1



        if self.mid_conv > 0:
            img_f_t = self.mid_conv_module(img_f_t)
        if self.last_hyper == 1:
            hyper_params = self.cls_output[:,cur_idx:]
            hyper_params = hyper_params.reshape(N * 3, self.feature_num, 1, 1)
            #hyper_params = self.tanh(hyper_params)
            hyper_params /= self.feature_num

            img_f_t = img_f_t.reshape(1, N * self.feature_num, H, W)
            out_img = F.conv2d(input=img_f_t, weight=hyper_params, groups=N)
            out_img = out_img.reshape(N,3,H,W)
        else:
            out_img = self.conv_out(img_f_t)



        #img_f = self.conv_emb(org_img)
        if self.residual == 1:
            org_img = org_img.reshape(N, 3, H, W)
            out_img = out_img + org_img

        if self.local_residual == 1:
            if self.div == 4:
                y1 = self.res1(out_img)
                y2 = self.res2(y1)
                y3 = self.res3(y2)
                y3 = self.res4(y3)
                y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
                y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
                y1 = self.res7(y1)
            elif self.div == 16:
                y1 = self.res1(out_img)
                y2 = self.res2(y1)
                y3 = self.res3(y2)
                y4 = self.res4(y3)
                y5 = self.res5(y4)
                y5 = self.res6(y5)
                y4 = self.res7(torch.cat((y4, F.upsample_bilinear(y5, scale_factor=2)), dim=1))
                y3 = self.res8(torch.cat((y3, F.upsample_bilinear(y4, scale_factor=2)), dim=1))
                y2 = self.res9(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
                y1 = self.res10(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
                y1 = self.res11(y1)
            elif self.div == 8:
                y1 = self.res1(out_img)
                y2 = self.res2(y1)
                y3 = self.res3(y2)
                y4 = self.res4(y3)
                y4 = self.res5(y4)
                y3 = self.res6(torch.cat((y3, F.upsample_bilinear(y4, scale_factor=2)), dim=1))
                y2 = self.res7(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
                y1 = self.res8(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
                y1 = self.res9(y1)

            out_img = y1 + out_img
        return out_img


class DCPNet28(nn.Module):
    def __init__(self, config):
        super(DCPNet28, self).__init__()

        self.control_point_num = config.control_point + 2
        self.feature_num = config.feature_num
        
        self.hyper = config.hyper
        self.hyper2 = config.hyper2
        self.xoffset = config.xoffset
        self.res_guide = config.res_guide
        if config.res_guide == 0:
            self.add_num = 0
        else:
            self.add_num = 256

        self.num_weight = config.num_weight

        param_num = self.control_point_num * self.feature_num * config.num_weight
        if self.hyper2 == 0:
            self.offset_params = nn.Parameter(torch.randn(1,self.control_point_num * self.feature_num) * 0.04) 
        if self.xoffset == 1:
            param_num += (self.control_point_num - 2) * self.feature_num
        if self.hyper == 0:    
            self.classifier = resnet18_224(out_dim=param_num, res_size=config.res_size, res_num=config.res_num)
            self.params = nn.Parameter(torch.randn(self.feature_num, 3, 1, 1))
        elif self.hyper == 1:
            param_num += (3 * self.feature_num)
            self.classifier = resnet18_224(out_dim=param_num, res_size=config.res_size, res_num=config.res_num)

        
        self.colorTransform = colorTransform3(self.control_point_num, config.offset_param, config)
        if config.conv_mode == 3:
            self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=3, stride=1, padding=1)
        elif config.conv_mode == 1:
            self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=1, stride=1, padding=0)
        
        
        self.act = 'leaky'
        if self.num_weight >= 0:
    
            self.resize = T.Resize((config.local_size, config.local_size))
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

            self.res5 = convBlock2(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)  # s5

            self.res6 = convBlock2(input_feature=256 + self.add_num, output_feature=256, ksize=3, stride=1, pad=1, act=self.act)  # s5

            # upsample and concat
            self.res7 = convBlock2(input_feature=384, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

            # upsample and concat
            self.res8 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

             # upsample and concat
            self.res9 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

            # upsample and concat
            #self.res10 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res10 = convBlock2(input_feature=48, output_feature=self.feature_num * 3, ksize=3, stride=1, pad=1, act=self.act)  # s1


            #self.res11 = nn.Conv2d(16, self.num_weight, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, org_img, index_image, color_map_control):
        N, C, H, W = org_img.shape
        self.cls_output = self.classifier(org_img)
        if self.num_weight >= 1:
            y = self.resize(org_img)
            y1 = self.res1(y)
            y2 = self.res2(y1)
            y3 = self.res3(y2)
            y4 = self.res4(y3)
            y5 = self.res5(y4)

            if self.res_guide == 1:
                _,_,nh,nw = y5.shape
                add_feat = self.cls_output[:,:self.add_num]
                add_feat = add_feat.reshape(N,self.add_num,1,1)
                add_feat = add_feat.repeat(1,1,nh,nw)
                y5 = torch.cat((y5,add_feat), dim=1)
            y5 = self.res6(y5)
            y4 = self.res7(torch.cat((y4, F.upsample_bilinear(y5, scale_factor=2)), dim=1))
            y3 = self.res8(torch.cat((y3, F.upsample_bilinear(y4, scale_factor=2)), dim=1))
            y2 = self.res9(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
            y = self.res10(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
            m = T.Resize((H, W))
            y = m(y)
            y = self.tanh(y)
            # y.

        if self.hyper == 0:
            norm_params = self.sigmoid(self.params)
            epsilon = 1e-10
            w_sum = torch.sum(norm_params, dim=1, keepdim=True)
            norm_params = norm_params / (w_sum + epsilon)
            # 64 x 3 x 1 x 1
            img_f = F.conv2d(input=org_img, weight=norm_params)
            if self.hyper2 == 0:
                self.offset_params_repeat = self.offset_params.repeat(N,1)
                img_f_t = self.colorTransform(img_f, self.offset_params_repeat, index_image, color_map_control)
            else:
                if self.num_weight == 1:
                    img_f_t = self.colorTransform(img_f, self.cls_output, index_image, color_map_control)
                else:
                    img_f_t = self.colorTransform(img_f, self.cls_output, index_image, color_map_control, y)
        elif self.hyper == 1:
            transform_params = self.cls_output[:,:3 * self.feature_num]
            offset_param = self.cls_output[:,3 * self.feature_num:]
            
            transform_params = transform_params.reshape(N * self.feature_num, 3)
            transform_params = self.sigmoid(transform_params)
            epsilon = 1e-10
            t_sum = torch.sum(transform_params, dim=1, keepdim=True)
            transform_params = transform_params / (t_sum + epsilon)
            transform_params = transform_params.reshape(N * self.feature_num, 3,1,1)
            #transform_params = transform_params.permute(1,2,0,3,4)
            #org_img = org_img.permute(1,0,2,3)
            org_img = org_img.reshape(1, N * 3, H, W)
            img_f = F.conv2d(input=org_img, weight=transform_params, groups=N)
            img_f = img_f.reshape(N,self.feature_num,H,W)
            if self.num_weight == 1:
                img_f_t = self.colorTransform(img_f, offset_param, index_image, color_map_control)
            else:
                img_f_t = self.colorTransform(img_f, offset_param, index_image, color_map_control, y)

        
        #self.conv_emb = F.conv2d(3, self.feature_num, weight=norm_params, kernel_size=1, stride=1, padding=0, bias=False)
        #self.temp_weight =
        #conv_emb = nn.Conv2d(3, self.feature_num, weight= , kernel_size=1, stride=1, padding=0, bias=False)

        img_f_t = img_f_t.repeat(1,3,1,1)
        img_f_t = img_f_t.reshape(N, self.feature_num, 3, H, W)
        y = y.reshape(N, self.feature_num, 3, H, W)
        out_img = img_f_t * y
        out_img = torch.sum(out_img, dim=1)
        #out_img = self.conv_out(img_f_t)

        #img_f = self.conv_emb(org_img)

        return out_img

class DCPNet29(nn.Module):
    def __init__(self, config, scale):
        super(DCPNet29, self).__init__()

        self.control_point_num = config.control_point + 2
        self.feature_num = config.feature_num
        self.scale = scale
        
        self.hyper = config.hyper
        self.xoffset = config.xoffset
        self.transform_num = config.transform_num
        self.conv_num = config.conv_num
        self.last_hyper = config.last_hyper

        self.residual = config.residual
        self.hyper_conv = config.hyper_conv

        self.local_residual = config.local_residual
        self.bias = config.bias
        self.quad = config.quad

        self.leaky_relu = nn.LeakyReLU(0.1)
        param_num = (self.control_point_num * self.feature_num) * self.transform_num
        param_num1 = param_num
        if self.bias == 1:
            param_num += (self.feature_num * self.transform_num)
        if self.quad == 1 or self.quad == 4 or self.quad == 5:
            param_num += (self.feature_num * self.transform_num)
        if self.quad == 2 or self.quad == 3 or self.quad == 6 or self.quad == 7:
            param_num += (self.feature_num * self.transform_num) * 2
        if self.last_hyper == 1:
            param_num += (3 * self.feature_num)
        param_xoffset = 0
        if self.xoffset == 1:
            param_xoffset = ((self.control_point_num - 2) * self.feature_num) * self.transform_num
            param_num += ((self.control_point_num - 2) * self.feature_num) * self.transform_num
        elif self.xoffset == 2:
            param_xoffset = ((self.control_point_num - 1) * self.feature_num) * self.transform_num
            param_num += ((self.control_point_num - 1) * self.feature_num) * self.transform_num

        if self.hyper == 0:
            if config.new_res == 0:
                self.classifier = resnet18_224(out_dim=param_num, res_size=config.res_size, res_num=config.res_num, fc_num=config.fc_num)
            elif config.new_res == 1:
                self.classifier = resnet18_224_2(out_dim=param_num, out_dim2=0, out_dim3=param_xoffset, res_size=config.res_size, res_num=config.res_num,
                                               fc_num=config.fc_num, init_w=config.init_w, init_w2=config.init_w2)
            self.params = nn.Parameter(torch.randn(self.feature_num, 3, 1, 1))
        elif self.hyper == 1:
            if self.hyper_conv == 1:
                param_num += (3 * self.feature_num)
                param_num2 = (3 * self.feature_num)
            elif self.hyper_conv == 3:
                param_num += (3 * self.feature_num) * 9
                param_num2 = (3 * self.feature_num) * 9
            if config.new_res == 0:
                self.classifier = resnet18_224(out_dim=param_num, res_size=config.res_size, res_num=config.res_num, fc_num=config.fc_num)
            elif config.new_res == 1:
                self.classifier = resnet18_224_2(out_dim=param_num1, out_dim2=param_num2, out_dim3=param_xoffset, res_size=config.res_size, res_num=config.res_num,
                                                 fc_num=config.fc_num, init_w=config.init_w, init_w2=config.init_w2)


        self.mid_conv = config.mid_conv
        conv_list = []
        for i in range(0, self.mid_conv):
            if config.mid_conv_mode == 'res':
                if config.mid_conv_size == 3:
                    conv_list.append(resBlock2(self.feature_num, self.feature_num, ksize=3, stride=1, pad=1, extra_conv=False, act='relu'))
                else:
                    conv_list.append(resBlock2(self.feature_num, self.feature_num, ksize=1, stride=1, pad=0, extra_conv=False, act='relu'))
            elif config.mid_conv_mode == 'res2':
                    conv_list.append(resBlock3(self.feature_num, self.feature_num, ksize=3, stride=1, pad=1, extra_conv=False, act='relu'))
            else:
                if config.mid_conv_size == 3:
                    conv_list.append(convBlock2(self.feature_num, self.feature_num, ksize=3, stride=1, pad=1, extra_conv=False, act='relu'))
                else:
                    conv_list.append(convBlock2(self.feature_num, self.feature_num, ksize=1, stride=1, pad=0, extra_conv=False, act='relu'))
        if self.mid_conv > 0:
            self.mid_conv_module = nn.Sequential(*conv_list)
        if config.xoffset == 0:
            self.colorTransform = colorTransform3(self.control_point_num, config.offset_param, config)
        elif config.xoffset == -1:
            self.colorTransform = colorTransform4(self.control_point_num, config.offset_param, config)
        else:
            self.colorTransform = colorTransform_xoffset(self.control_point_num, config.offset_param, config)
        if config.conv_mode == 3:
            self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=3, stride=1, padding=1)
        elif config.conv_mode == 1:
            self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=1, stride=1, padding=0)

        self.act = 'relu'
        self.pixelwise_multi = config.pixelwise_multi
        if self.pixelwise_multi == 1 or self.pixelwise_multi == 3 or self.pixelwise_multi == 4 or self.pixelwise_multi == 6:
            self.resize = T.Resize((config.local_size // (2**scale), config.local_size // (2**scale)))
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

            self.res5 = convBlock2(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)  # s5

            self.res6 = convBlock2(input_feature=256, output_feature=256, ksize=3, stride=1, pad=1, act=self.act)  # s5

            # upsample and concat
            self.res7 = convBlock2(input_feature=384, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

            # upsample and concat
            self.res8 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

             # upsample and concat
            self.res9 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

            # upsample and concat
            self.res10 = convBlock2(input_feature=48, output_feature=self.feature_num, ksize=3, stride=1, pad=1, act=self.act)  # s1

            self.res11 = nn.Conv2d(self.feature_num, self.feature_num, kernel_size=3, padding=1)
        
        if self.pixelwise_multi == 2 or self.pixelwise_multi == 5:
            self.resize = T.Resize((config.local_size // (2**scale), config.local_size // (2**scale)))
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

            self.res5 = convBlock2(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)  # s5

            self.res6 = convBlock2(input_feature=256, output_feature=256, ksize=3, stride=1, pad=1, act=self.act)  # s5

            # upsample and concat
            self.res7 = convBlock2(input_feature=384, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

            # upsample and concat
            self.res8 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

             # upsample and concat
            self.res9 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

            # upsample and concat
            self.res10 = convBlock2(input_feature=48, output_feature=self.feature_num * 2, ksize=3, stride=1, pad=1, act=self.act)  # s1

            self.res11 = nn.Conv2d(self.feature_num * 2, self.feature_num * 2, kernel_size=3, padding=1)

        self.div = config.div
        if self.local_residual == 1:
            if self.div == 4:
                self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
                self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
                self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

                self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
                # upsample and concat
                self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
                # upsample and concat
                self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
                self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True, act=self.act)
            elif self.div == 16:
                self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
                self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
                self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

                self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

                self.res5 = convBlock2(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)  # s5

                self.res6 = convBlock2(input_feature=256, output_feature=256, ksize=3, stride=1, pad=1, act=self.act)  # s5

                # upsample and concat
                self.res7 = convBlock2(input_feature=384, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

                # upsample and concat
                self.res8 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

                 # upsample and concat
                self.res9 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

                # upsample and concat
                self.res10 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1

                self.res11 = nn.Conv2d(16, 3, kernel_size=3, padding=1)

            elif self.div == 8:
                self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
                self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
                self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

                self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

                self.res5 = convBlock2(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

                # upsample and concat
                self.res6 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

                 # upsample and concat
                self.res7 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

                # upsample and concat
                self.res8 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1

                self.res9 = nn.Conv2d(16, 3, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.act_mode = config.act_mode

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, org_img, index_image, color_map_control):
        N, C, H, W = org_img.shape
        m = T.Resize((H // (2**self.scale), W // (2**self.scale)))
        org_img = m(org_img)
        N, C, H, W = org_img.shape
        self.cls_output = self.classifier(org_img)
        if self.pixelwise_multi >= 1:
            y = self.resize(org_img)
            y1 = self.res1(y)
            y2 = self.res2(y1)
            y3 = self.res3(y2)
            y4 = self.res4(y3)
            y5 = self.res5(y4)
            y5 = self.res6(y5)
            y4 = self.res7(torch.cat((y4, F.upsample_bilinear(y5, scale_factor=2)), dim=1))
            y3 = self.res8(torch.cat((y3, F.upsample_bilinear(y4, scale_factor=2)), dim=1))
            y2 = self.res9(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
            y1 = self.res10(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
            y1 = self.res11(y1)
            m = T.Resize((H, W))
            y = m(y1)
        if self.hyper == 0:
            if self.act_mode == 'sigmoid':
                norm_params = self.sigmoid(self.params)
                epsilon = 1e-10
                w_sum = torch.sum(torch.abs(norm_params), dim=1, keepdim=True)
                norm_params = norm_params / (w_sum + epsilon)
            elif self.act_mode == 'tanh':
                norm_params = self.tanh(self.params)
                epsilon = 1e-10
                w_norm = torch.norm(norm_params, dim=1, keepdim=True)
                norm_params = norm_params / (w_norm + epsilon)

            # 64 x 3 x 1 x 1
            img_f = F.conv2d(input=org_img, weight=norm_params)
            if self.act_mode == 'tanh':
                img_f = (img_f + 1) / 2
            img_f_t = self.colorTransform(img_f, self.cls_output, index_image, color_map_control)
            if self.act_mode == 'tanh':
                img_f_t = img_f_t * 2 - 1
        elif self.hyper == 1:
            N, C, H, W = org_img.shape
            if self.hyper_conv == 1:
                cur_idx = 3 * self.feature_num
            elif self.hyper_conv == 3:
                cur_idx = 3 * self.feature_num * 9
            transform_params = self.cls_output[:,:cur_idx]
            if self.bias == 1:
                bias_params = self.cls_output[:,cur_idx:cur_idx + self.feature_num]
                cur_idx += self.feature_num
            if self.hyper_conv == 1:
                transform_params = transform_params.reshape(N * self.feature_num, 3)
            elif self.hyper_conv == 3:
                transform_params = transform_params.reshape(N * self.feature_num, 27)
            if self.act_mode == 'sigmoid':
                transform_params = self.sigmoid(transform_params)
                epsilon = 1e-10
                t_sum = torch.sum(transform_params, dim=1, keepdim=True)
                transform_params = transform_params / (t_sum + epsilon)
            elif self.act_mode == 'tanh':
                transform_params = self.tanh(transform_params)
                epsilon = 1e-10
                w_norm = torch.sum(torch.abs(transform_params), dim=1, keepdim=True)
                transform_params = transform_params / (w_norm + epsilon)
            elif self.act_mode == 'minmax' or self.act_mode == 'trunc' or self.act_mode == 'minmax2':
                transform_params = transform_params
            elif self.act_mode == 'minmax3':
                transform_params = self.sigmoid(transform_params)
                epsilon = 1e-10
                t_sum = torch.sum(transform_params, dim=1, keepdim=True)
                transform_params = transform_params / (t_sum + epsilon)

            if self.hyper_conv == 1:
                transform_params = transform_params.reshape(N * self.feature_num, 3, 1, 1)
            elif self.hyper_conv == 3:
                transform_params = transform_params.reshape(N * self.feature_num, 3, 3, 3)
            #transform_params = transform_params.permute(1,2,0,3,4)
            #org_img = org_img.permute(1,0,2,3)
            org_img = org_img.reshape(1, N * 3, H, W)
            if self.bias == 1:
                bias_params = bias_params.reshape(N * self.feature_num)
                if self.hyper_conv == 1:
                    img_f = F.conv2d(input=org_img, weight=transform_params, bias=bias_params, groups=N)
                elif self.hyper_conv == 3:
                    img_f = F.conv2d(input=org_img, weight=transform_params, bias=bias_params, groups=N, padding=1)
            else:
                if self.hyper_conv == 1:
                    img_f = F.conv2d(input=org_img, weight=transform_params, groups=N)
                elif self.hyper_conv == 3:
                    img_f = F.conv2d(input=org_img, weight=transform_params, groups=N, padding=1)

            img_f = img_f.reshape(N,self.feature_num,H,W)

            if self.quad == 1:
                quad_params = self.tanh(self.cls_output[:,cur_idx:cur_idx + self.feature_num])
                cur_idx += self.feature_num

                quad_params = quad_params.reshape(N, self.feature_num, 1, 1)
                img_f = img_f + quad_params * img_f * (1.0 - img_f)
            
            elif self.quad == 6 or self.quad == 7:
                gamma_params = self.cls_output[:,cur_idx:cur_idx + self.feature_num]
                cur_idx += self.feature_num
                beta_params = self.cls_output[:,cur_idx:cur_idx + self.feature_num]
                cur_idx += self.feature_num

                gamma_params = gamma_params.reshape(N, self.feature_num, 1, 1)
                beta_params = beta_params.reshape(N, self.feature_num, 1, 1)
                if self.quad == 6:
                    img_f = img_f * gamma_params + beta_params
                elif self.quad == 7:
                    img_f = img_f * gamma_params

            if self.pixelwise_multi == 1:
                img_f = img_f * y
            elif self.pixelwise_multi == 2:
                img_f = img_f * y[:,:self.feature_num,:,:] + y[:,self.feature_num:,:,:]
            elif self.pixelwise_multi == 3:
                y = self.tanh(y)
                img_f = img_f + y * img_f * ( 1.0 - img_f)
            if self.act_mode == 'minmax' or self.act_mode == 'minmax2' or self.act_mode == 'minmax3':
                    # normalize
                min_val, _ = img_f.min(dim=2, keepdim=True)
                min_val, _ = min_val.min(dim=3, keepdim=True)

                max_val, _ = img_f.max(dim=2, keepdim=True)
                max_val, _ = max_val.max(dim=3, keepdim=True)
                img_f = (img_f - min_val) / (max_val - min_val)
            elif self.act_mode == 'trunc':
                img_f = torch.clamp(img_f, 0, 1)

            if self.act_mode == 'tanh':
                img_f = (img_f + 1.0) / 2
                img_f = torch.clamp(img_f, 0, 1)

            for i in range(0,self.transform_num):
                plus_idx = self.control_point_num * self.feature_num
                if self.xoffset == 1:
                    plus_idx += ((self.control_point_num - 2) * self.feature_num)
                offset_param = self.cls_output[:,cur_idx:cur_idx + plus_idx]
                cur_idx += plus_idx
                img_f_t = self.colorTransform(img_f, offset_param, index_image, color_map_control)
                if self.act_mode == 'minmax2':
                    img_f_t = min_val + (max_val - min_val) * img_f_t
                if i < self.transform_num - 1:
                    # normalize
                    min_val, _ = img_f_t.min(dim=2, keepdim=True)
                    min_val, _ = min_val.min(dim=3, keepdim=True)

                    max_val, _ = img_f_t.max(dim=2, keepdim=True)
                    max_val, _ = max_val.max(dim=3, keepdim=True)
                    img_f_t = (img_f_t - min_val) / (max_val - min_val)
                    if self.act_mode == 'tanh':
                        img_f_t = img_f_t * 2.0 - 1

        if self.pixelwise_multi == 4:
            img_f_t = img_f_t * y
        elif self.pixelwise_multi == 5:
            img_f_t = img_f_t * y[:,:self.feature_num,:,:] + y[:,self.feature_num:,:,:]
        elif self.pixelwise_multi == 6:
            y = self.tanh(y)
            img_f_t = img_f_t + y * img_f_t * ( 1.0 - img_f_t)

        return img_f_t

class DCPNet29_cor(nn.Module):
    def __init__(self, config):
        super(DCPNet29_cor, self).__init__()
        self.feature_num = config.feature_num
        self.upsample_mode = config.upsample_mode
        self.model1 = DCPNet29(config, 0)
        self.model2 = DCPNet29(config, 1)
        self.model3 = DCPNet29(config, 2)
        if self.upsample_mode == 1:
            self.conv_out = nn.Conv2d(self.feature_num * 3, 3, kernel_size=1, stride=1, padding=0)
        elif self.upsample_mode == 2:
            self.conv_out1 = nn.Conv2d(self.feature_num * 2, self.feature_num, kernel_size=1, stride=1, padding=0)
            self.bn = nn.BatchNorm2d(self.feature_num)
            self.relu = nn.ReLU()
            self.conv_out2 = nn.Conv2d(self.feature_num * 2, self.feature_num, kernel_size=1, stride=1, padding=0)
            self.bn1 = nn.BatchNorm2d(self.feature_num)
            self.relu1 = nn.ReLU()
            self.conv_out3 = nn.Conv2d(self.feature_num, 3, kernel_size=1, stride=1, padding=0)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, org_img, index_image, color_map_control):
        feat1 = self.model1(org_img,index_image, color_map_control)
        feat2 = self.model2(org_img,index_image, color_map_control)
        feat3 = self.model3(org_img,index_image, color_map_control)
        if self.upsample_mode == 1:
            feat3 = F.upsample_bilinear(feat3, scale_factor=4) 
            feat2 = F.upsample_bilinear(feat2, scale_factor=2) 
            feat = torch.cat((feat1,feat2,feat3), dim=1)
            out_img = self.conv_out(feat)
        elif self.upsample_mode == 2:
            feat3 = F.upsample_bilinear(feat3, scale_factor=2)
            feat2 = torch.cat((feat2,feat3), dim=1)
            feat2 = self.conv_out1(feat2)
            feat2 = self.bn(feat2)
            feat2 = self.relu(feat2)
            feat2 = F.upsample_bilinear(feat2, scale_factor=2)
            feat1 = torch.cat((feat1,feat2), dim=1)
            feat1 = self.conv_out2(feat1)
            feat1 = self.bn1(feat1)
            feat1 = self.relu1(feat1)
            out_img = self.conv_out3(feat1)
            
        return out_img


class DCPNet30(nn.Module):
    def __init__(self, config):
        super(DCPNet30, self).__init__()

        self.control_point_num = config.control_point + 2
        self.feature_num = config.feature_num

        self.hyper = config.hyper
        self.xoffset = config.xoffset
        self.transform_num = config.transform_num
        self.conv_num = config.conv_num
        self.last_hyper = config.last_hyper

        self.residual = config.residual
        self.hyper_conv = config.hyper_conv

        self.local_residual = config.local_residual
        self.bias = config.bias
        self.quad = config.quad

        self.leaky_relu = nn.LeakyReLU(0.1)
        param_num = (self.control_point_num * self.feature_num) * self.transform_num
        if self.bias == 1:
            param_num += (self.feature_num * self.transform_num)
        if self.quad == 1 or self.quad == 4 or self.quad == 5:
            param_num += (self.feature_num * self.transform_num)
        if self.quad == 2 or self.quad == 3 or self.quad == 6 or self.quad == 7:
            param_num += (self.feature_num * self.transform_num) * 2
        if self.last_hyper == 1:
            param_num += (3 * self.feature_num)
        if self.xoffset == 1:
            param_num += ((self.control_point_num - 2) * self.feature_num) * self.transform_num

        self.mid_conv = config.mid_conv
        conv_list = []
        for i in range(0, self.mid_conv):
            if config.mid_conv_mode == 'res':
                if config.mid_conv_size == 3:
                    conv_list.append(
                        resBlock2(self.feature_num, self.feature_num, ksize=3, stride=1, pad=1, extra_conv=False,
                                  act='relu'))
                else:
                    conv_list.append(
                        resBlock2(self.feature_num, self.feature_num, ksize=1, stride=1, pad=0, extra_conv=False,
                                  act='relu'))
            elif config.mid_conv_mode == 'res2':
                conv_list.append(
                    resBlock3(self.feature_num, self.feature_num, ksize=3, stride=1, pad=1, extra_conv=False,
                              act='relu'))
            else:
                if config.mid_conv_size == 3:
                    conv_list.append(
                        convBlock2(self.feature_num, self.feature_num, ksize=3, stride=1, pad=1, extra_conv=False,
                                   act='relu'))
                else:
                    conv_list.append(
                        convBlock2(self.feature_num, self.feature_num, ksize=1, stride=1, pad=0, extra_conv=False,
                                   act='relu'))
        if self.mid_conv > 0:
            self.mid_conv_module = nn.Sequential(*conv_list)
        if config.xoffset == 0:
            self.colorTransform = colorTransform6(self.control_point_num, config.offset_param, config)
        elif config.xoffset == -1:
            self.colorTransform = colorTransform4(self.control_point_num, config.offset_param, config)
        elif config.xoffset == -2:
            self.colorTransform = colorTransform5(self.control_point_num, config.offset_param, config)
        elif config.xoffset == -3:
            self.colorTransform = colorTransform7(self.control_point_num, config.offset_param, config)
        else:
            self.colorTransform = colorTransform_xoffset(self.control_point_num, config.offset_param, config)
        if config.conv_mode == 3:
            self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=3, stride=1, padding=1).cuda(config.rank)
        elif config.conv_mode == 1:
            self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=1, stride=1, padding=0).cuda(config.rank)

        self.act = 'relu'
        self.pixelwise_multi = config.pixelwise_multi
        if self.pixelwise_multi == 1 or self.pixelwise_multi == 3 or self.pixelwise_multi == 4 or self.pixelwise_multi == 6:
            self.resize = T.Resize((config.local_size, config.local_size))
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

            self.res5 = convBlock2(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)  # s5

            self.res6 = convBlock2(input_feature=256, output_feature=256, ksize=3, stride=1, pad=1, act=self.act)  # s5

            # upsample and concat
            self.res7 = convBlock2(input_feature=384, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

            # upsample and concat
            self.res8 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

            # upsample and concat
            self.res9 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

            # upsample and concat
            self.res10 = convBlock2(input_feature=48, output_feature=self.feature_num, ksize=3, stride=1, pad=1,
                                    act=self.act)  # s1

            self.res11 = nn.Conv2d(self.feature_num, self.feature_num, kernel_size=3, padding=1)

        if self.pixelwise_multi == 2 or self.pixelwise_multi == 5:
            self.resize = T.Resize((config.local_size, config.local_size))
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)  # s4

            self.res5 = convBlock2(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)  # s5

            self.res6 = convBlock2(input_feature=256, output_feature=256, ksize=3, stride=1, pad=1, act=self.act)  # s5

            # upsample and concat
            self.res7 = convBlock2(input_feature=384, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)  # s4

            # upsample and concat
            self.res8 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3

            # upsample and concat
            self.res9 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2

            # upsample and concat
            self.res10 = convBlock2(input_feature=48, output_feature=self.feature_num * 2, ksize=3, stride=1, pad=1,
                                    act=self.act)  # s1

            self.res11 = nn.Conv2d(self.feature_num * 2, self.feature_num * 2, kernel_size=3, padding=1)

        self.div = config.div
        if self.local_residual == 1:
            if self.div == 4:
                self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
                self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1,
                                       act=self.act)  # s2
                self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1,
                                       act=self.act)  # s3

                self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1,
                                       act=self.act)  # s3
                # upsample and concat
                self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1,
                                       act=self.act)  # s2
                # upsample and concat
                self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1,
                                       act=self.act)  # s1
                self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                      act=self.act)
            elif self.div == 16:
                self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
                self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1,
                                       act=self.act)  # s2
                self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1,
                                       act=self.act)  # s3

                self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1,
                                       act=self.act)  # s4

                self.res5 = convBlock2(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1,
                                       act=self.act)  # s5

                self.res6 = convBlock2(input_feature=256, output_feature=256, ksize=3, stride=1, pad=1,
                                       act=self.act)  # s5

                # upsample and concat
                self.res7 = convBlock2(input_feature=384, output_feature=128, ksize=3, stride=1, pad=1,
                                       act=self.act)  # s4

                # upsample and concat
                self.res8 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1,
                                       act=self.act)  # s3

                # upsample and concat
                self.res9 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1,
                                       act=self.act)  # s2

                # upsample and concat
                self.res10 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1,
                                        act=self.act)  # s1

                self.res11 = nn.Conv2d(16, 3, kernel_size=3, padding=1)

            elif self.div == 8:
                self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
                self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1,
                                       act=self.act)  # s2
                self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1,
                                       act=self.act)  # s3

                self.res4 = convBlock2(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1,
                                       act=self.act)  # s4

                self.res5 = convBlock2(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1,
                                       act=self.act)  # s4

                # upsample and concat
                self.res6 = convBlock2(input_feature=192, output_feature=64, ksize=3, stride=1, pad=1,
                                       act=self.act)  # s3

                # upsample and concat
                self.res7 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1,
                                       act=self.act)  # s2

                # upsample and concat
                self.res8 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1,
                                       act=self.act)  # s1

                self.res9 = nn.Conv2d(16, 3, kernel_size=3, padding=1)

        if config.init_w > 0:
            self.initialize_weights()
        if self.hyper == 0:
            self.classifier = resnet18_224(out_dim=param_num, res_size=config.res_size, res_num=config.res_num,
                                           fc_num=config.fc_num, init_w=config.init_w)
            self.params = nn.Parameter(torch.randn(self.feature_num, 3, 1, 1))
        elif self.hyper == 1:
            if self.hyper_conv == 1:
                param_num += (3 * self.feature_num)
            elif self.hyper_conv == 3:
                param_num += (3 * self.feature_num) * 9
            self.classifier = resnet18_224(out_dim=param_num, res_size=config.res_size, res_num=config.res_num,
                                           fc_num=config.fc_num, init_w=config.init_w)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.act_mode = config.act_mode


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, org_img, index_image, color_map_control):
        N, C, H, W = org_img.shape
        self.cls_output = self.classifier(org_img)
        if self.pixelwise_multi >= 1:
            y = self.resize(org_img)
            y1 = self.res1(y)
            y2 = self.res2(y1)
            y3 = self.res3(y2)
            y4 = self.res4(y3)
            y5 = self.res5(y4)
            y5 = self.res6(y5)
            y4 = self.res7(torch.cat((y4, F.upsample_bilinear(y5, scale_factor=2)), dim=1))
            y3 = self.res8(torch.cat((y3, F.upsample_bilinear(y4, scale_factor=2)), dim=1))
            y2 = self.res9(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
            y1 = self.res10(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
            y1 = self.res11(y1)
            m = T.Resize((H, W))
            y = m(y1)
        if self.hyper == 0:
            if self.act_mode == 'sigmoid':
                norm_params = self.sigmoid(self.params)
                epsilon = 1e-10
                w_sum = torch.sum(torch.abs(norm_params), dim=1, keepdim=True)
                norm_params = norm_params / (w_sum + epsilon)
            elif self.act_mode == 'tanh':
                norm_params = self.tanh(self.params)
                epsilon = 1e-10
                w_norm = torch.norm(norm_params, dim=1, keepdim=True)
                norm_params = norm_params / (w_norm + epsilon)

            # 64 x 3 x 1 x 1
            img_f = F.conv2d(input=org_img, weight=norm_params)
            if self.act_mode == 'tanh':
                img_f = (img_f + 1) / 2
            img_f_t = self.colorTransform(img_f, self.cls_output, index_image, color_map_control)
            if self.act_mode == 'tanh':
                img_f_t = img_f_t * 2 - 1
        elif self.hyper == 1:
            N, C, H, W = org_img.shape
            if self.hyper_conv == 1:
                cur_idx = 3 * self.feature_num
            elif self.hyper_conv == 3:
                cur_idx = 3 * self.feature_num * 9
            transform_params = self.cls_output[:, :cur_idx]
            if self.bias == 1:
                bias_params = self.cls_output[:, cur_idx:cur_idx + self.feature_num]
                cur_idx += self.feature_num
            if self.hyper_conv == 1:
                transform_params = transform_params.reshape(N * self.feature_num, 3)
            elif self.hyper_conv == 3:
                transform_params = transform_params.reshape(N * self.feature_num, 27)
            if self.act_mode == 'sigmoid':
                transform_params = self.sigmoid(transform_params)
                epsilon = 1e-10
                t_sum = torch.sum(transform_params, dim=1, keepdim=True)
                transform_params = transform_params / (t_sum + epsilon)
            elif self.act_mode == 'tanh':
                transform_params = self.tanh(transform_params)
                epsilon = 1e-10
                w_norm = torch.sum(torch.abs(transform_params), dim=1, keepdim=True)
                transform_params = transform_params / (w_norm + epsilon)
            elif self.act_mode == 'minmax' or self.act_mode == 'trunc' or self.act_mode == 'minmax2':
                transform_params = transform_params
            elif self.act_mode == 'minmax3':
                transform_params = self.sigmoid(transform_params)
                epsilon = 1e-10
                t_sum = torch.sum(transform_params, dim=1, keepdim=True)
                transform_params = transform_params / (t_sum + epsilon)

            if self.hyper_conv == 1:
                transform_params = transform_params.reshape(N * self.feature_num, 3, 1, 1)
            elif self.hyper_conv == 3:
                transform_params = transform_params.reshape(N * self.feature_num, 3, 3, 3)
            # transform_params = transform_params.permute(1,2,0,3,4)
            # org_img = org_img.permute(1,0,2,3)
            org_img = org_img.reshape(1, N * 3, H, W)
            if self.bias == 1:
                bias_params = bias_params.reshape(N * self.feature_num)
                if self.hyper_conv == 1:
                    img_f = F.conv2d(input=org_img, weight=transform_params, bias=bias_params, groups=N)
                elif self.hyper_conv == 3:
                    img_f = F.conv2d(input=org_img, weight=transform_params, bias=bias_params, groups=N, padding=1)
            else:
                if self.hyper_conv == 1:
                    img_f = F.conv2d(input=org_img, weight=transform_params, groups=N)
                elif self.hyper_conv == 3:
                    img_f = F.conv2d(input=org_img, weight=transform_params, groups=N, padding=1)

            img_f = img_f.reshape(N, self.feature_num, H, W)

            if self.quad == 1:
                quad_params = self.tanh(self.cls_output[:, cur_idx:cur_idx + self.feature_num])
                cur_idx += self.feature_num

                quad_params = quad_params.reshape(N, self.feature_num, 1, 1)
                img_f = img_f + quad_params * img_f * (1.0 - img_f)

            elif self.quad == 6 or self.quad == 7:
                gamma_params = self.cls_output[:, cur_idx:cur_idx + self.feature_num]
                cur_idx += self.feature_num
                beta_params = self.cls_output[:, cur_idx:cur_idx + self.feature_num]
                cur_idx += self.feature_num

                gamma_params = gamma_params.reshape(N, self.feature_num, 1, 1)
                beta_params = beta_params.reshape(N, self.feature_num, 1, 1)
                if self.quad == 6:
                    img_f = img_f * gamma_params + beta_params
                elif self.quad == 7:
                    img_f = img_f * gamma_params

            if self.pixelwise_multi == 1:
                img_f = img_f * y
            elif self.pixelwise_multi == 2:
                img_f = img_f * y[:, :self.feature_num, :, :] + y[:, self.feature_num:, :, :]
            elif self.pixelwise_multi == 3:
                y = self.tanh(y)
                img_f = img_f + y * img_f * (1.0 - img_f)
            if self.act_mode == 'minmax' or self.act_mode == 'minmax2' or self.act_mode == 'minmax3':
                # normalize
                min_val, _ = img_f.min(dim=2, keepdim=True)
                min_val, _ = min_val.min(dim=3, keepdim=True)

                max_val, _ = img_f.max(dim=2, keepdim=True)
                max_val, _ = max_val.max(dim=3, keepdim=True)
                img_f = (img_f - min_val) / (max_val - min_val)
            elif self.act_mode == 'trunc':
                img_f = torch.clamp(img_f, 0, 1)

            if self.act_mode == 'tanh':
                img_f = (img_f + 1.0) / 2
                img_f = torch.clamp(img_f, 0, 1)

            for i in range(0, self.transform_num):
                plus_idx = self.control_point_num * self.feature_num
                if self.xoffset == 1:
                    plus_idx += ((self.control_point_num - 2) * self.feature_num)
                offset_param = self.cls_output[:, cur_idx:cur_idx + plus_idx]
                cur_idx += plus_idx
                img_f_t, params = self.colorTransform(img_f, offset_param, index_image, color_map_control)
                if self.act_mode == 'minmax2':
                    img_f_t = min_val + (max_val - min_val) * img_f_t
                if i < self.transform_num - 1:
                    # normalize
                    min_val, _ = img_f_t.min(dim=2, keepdim=True)
                    min_val, _ = min_val.min(dim=3, keepdim=True)

                    max_val, _ = img_f_t.max(dim=2, keepdim=True)
                    max_val, _ = max_val.max(dim=3, keepdim=True)
                    img_f_t = (img_f_t - min_val) / (max_val - min_val)
                    if self.act_mode == 'tanh':
                        img_f_t = img_f_t * 2.0 - 1

        if self.pixelwise_multi == 4:
            img_f_t = img_f_t * y
        elif self.pixelwise_multi == 5:
            img_f_t = img_f_t * y[:, :self.feature_num, :, :] + y[:, self.feature_num:, :, :]
        elif self.pixelwise_multi == 6:
            y = self.tanh(y)
            img_f_t = img_f_t + y * img_f_t * (1.0 - img_f_t)

        if self.mid_conv > 0:
            img_f_t = self.mid_conv_module(img_f_t)
        if self.last_hyper == 1:
            hyper_params = self.cls_output[:, cur_idx:]
            hyper_params = hyper_params.reshape(N * 3, self.feature_num, 1, 1)
            # hyper_params = self.tanh(hyper_params)
            hyper_params /= self.feature_num

            img_f_t = img_f_t.reshape(1, N * self.feature_num, H, W)
            out_img = F.conv2d(input=img_f_t, weight=hyper_params, groups=N)
            out_img = out_img.reshape(N, 3, H, W)
        else:
            out_img = self.conv_out(img_f_t)

        if self.quad == 2 or self.quad == 3:
            gamma_params = self.cls_output[:, cur_idx:cur_idx + self.feature_num]
            cur_idx += self.feature_num
            beta_params = self.cls_output[:, cur_idx:cur_idx + self.feature_num]
            cur_idx += self.feature_num

            gamma_params = gamma_params.reshape(N, self.feature_num, 1, 1)
            beta_params = beta_params.reshape(N, self.feature_num, 1, 1)
            img_f_t = img_f_t * gamma_params + beta_params
            if self.quad == 3:
                img_f_t = self.relu(img_f_t)
        elif self.quad == 4:
            quad_params = self.tanh(self.cls_output[:, cur_idx:cur_idx + self.feature_num])
            cur_idx += self.feature_num
            quad_params = quad_params.reshape(N, self.feature_num, 1, 1)
            img_f_t = img_f_t + quad_params * img_f_t * (1.0 - img_f_t)
        elif self.quad == 5:
            gamma_params = self.sigmoid(self.cls_output[:, cur_idx:cur_idx + self.feature_num])
            cur_idx += self.feature_num
            gamma_params = gamma_params.reshape(N, self.feature_num, 1, 1)
            img_f_t = img_f_t * gamma_params

        # img_f = self.conv_emb(org_img)
        if self.residual == 1:
            org_img = org_img.reshape(N, 3, H, W)
            out_img = out_img + org_img

        if self.local_residual == 1:
            if self.div == 4:
                y1 = self.res1(out_img)
                y2 = self.res2(y1)
                y3 = self.res3(y2)
                y3 = self.res4(y3)
                y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
                y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
                y1 = self.res7(y1)
            elif self.div == 16:
                y1 = self.res1(out_img)
                y2 = self.res2(y1)
                y3 = self.res3(y2)
                y4 = self.res4(y3)
                y5 = self.res5(y4)
                y5 = self.res6(y5)
                y4 = self.res7(torch.cat((y4, F.upsample_bilinear(y5, scale_factor=2)), dim=1))
                y3 = self.res8(torch.cat((y3, F.upsample_bilinear(y4, scale_factor=2)), dim=1))
                y2 = self.res9(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
                y1 = self.res10(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
                y1 = self.res11(y1)
            elif self.div == 8:
                y1 = self.res1(out_img)
                y2 = self.res2(y1)
                y3 = self.res3(y2)
                y4 = self.res4(y3)
                y4 = self.res5(y4)
                y3 = self.res6(torch.cat((y3, F.upsample_bilinear(y4, scale_factor=2)), dim=1))
                y2 = self.res7(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
                y1 = self.res8(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
                y1 = self.res9(y1)

            out_img = y1 + out_img
        return out_img, params

class colorTransform3(nn.Module):
    def __init__(self, control_point=16, offset_param=0.04, config=0):
        super(colorTransform3, self).__init__()
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)
        self.control_point = control_point
        self.config = config
        self.feature_num = config.feature_num

        self.epsilon = 1e-8
        
        self.offset_param = offset_param
        if self.offset_param != -1:
            if config.trainable_offset == 1:
                self.offset_param = nn.Parameter(torch.tensor([offset_param], dtype=torch.float32))
            else:
                self.offset_param = offset_param

    def forward(self, org_img, params, color_mapping_global_a, color_map_control):
        N, C, H, W = org_img.shape
        color_map_control_x = color_map_control.clone()
        if self.offset_param != -1:
            params = params.reshape(N, self.feature_num, self.control_point) * self.offset_param
        else:
            params = params.reshape(N, self.feature_num, self.control_point)
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


class colorTransform_no(nn.Module):
    def __init__(self, control_point=16, offset_param=0.04, config=0):
        super(colorTransform_no, self).__init__()
        self.config = config

    def forward(self, org_img, params, color_mapping_global_a, color_map_control):
        return org_img

class colorTransform31(nn.Module):
    def __init__(self, control_point=16, offset_param=0.04, config=0):
        super(colorTransform31, self).__init__()
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)
        self.control_point = control_point
        self.config = config
        self.feature_num = config.feature_num

        self.epsilon = 1e-8
        
        self.offset_param = offset_param
        if self.offset_param != -1:
            if config.trainable_offset == 1:
                self.offset_param = nn.Parameter(torch.tensor([offset_param], dtype=torch.float32))
            else:
                self.offset_param = offset_param

    def forward(self, org_img, params, color_mapping_global_a, color_map_control):
        N, C, H, W = org_img.shape
        color_map_control_x = color_map_control.clone()
        if self.offset_param != -1:
            params = params.reshape(N, self.feature_num, self.control_point) * self.offset_param
        else:
            params = params.reshape(N, self.feature_num, self.control_point)
        color_map_control_y = color_map_control_x + params

        color_map_control_y = torch.cat((color_map_control_y, color_map_control_y[:, :, self.control_point-1:self.control_point]), dim=2)
        #color_map_control_x = torch.cat((color_map_control_x, color_map_control_x[:, :, self.control_point-1:self.control_point]), dim=2)
        img_reshaped = org_img.reshape(N, self.feature_num, -1)
        #img_reshaped_val = img_reshaped * (self.control_point-1)


        img_reshaped_index = torch.floor(img_reshaped * (self.control_point-1))
        img_reshaped_index = img_reshaped_index.type(torch.int64)
        #img_reshaped_index_plus = img_reshaped_index + 1

        #img_reshaped_val
        img_reshaped_coeff = (img_reshaped * (self.control_point-1)) - img_reshaped_index
        #img_reshaped_coeff_one = 1.0 - img_reshaped_coeff

        mapped_color_map_control_y = torch.gather(color_map_control_y, 2, img_reshaped_index)
        #img_reshaped_index_plus
        mapped_color_map_control_y_plus = torch.gather(color_map_control_y, 2, img_reshaped_index + 1)

        mapped_color_map_control_y = (1.0 - img_reshaped_coeff) * mapped_color_map_control_y + img_reshaped_coeff * mapped_color_map_control_y_plus

        mapped_color_map_control_y = mapped_color_map_control_y.reshape(N, C, H, W)
        return mapped_color_map_control_y
    
class colorTransform4(nn.Module):
    def __init__(self, control_point=16, offset_param=0.04, config=0):
        super(colorTransform4, self).__init__()
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)
        self.control_point = control_point
        self.sigmoid = torch.nn.Sigmoid()
        self.config = config
        self.feature_num = config.feature_num

        self.epsilon = 1e-8

        self.offset_param = offset_param
        if self.offset_param > 0:
            self.offset_param_val = nn.Parameter(torch.tensor([offset_param], dtype=torch.float32))
        
        



    def forward(self, org_img, params, color_mapping_global_a, color_map_control):
        N, C, H, W = org_img.shape
        #out_img = torch.zeros_like(org_img)
        color_map_control_x = color_map_control.clone()
        params = params.reshape(N, self.feature_num, self.control_point)
        if self.offset_param == -1:
            color_map_control_y = params
        elif self.offset_param == 0:
            color_map_control_y = self.sigmoid(params)
        else:
            color_map_control_y = params * self.offset_param_val

        color_map_control_y = torch.cat((color_map_control_y, color_map_control_y[:, :, self.control_point-1:self.control_point]), dim=2)
        color_map_control_x = torch.cat((color_map_control_x, color_map_control_x[:, :, self.control_point-1:self.control_point]), dim=2)
        img_reshaped = org_img.reshape(N, self.feature_num, -1)
        #out_img_reshaped = out_img.reshape(N, self.feature_num, -1)
        img_reshaped_val = img_reshaped * (self.control_point-1)


        img_reshaped_index = torch.floor(img_reshaped * (self.control_point-1))
        img_reshaped_index = img_reshaped_index.type(torch.int64)
        img_reshaped_index_plus = img_reshaped_index + 1

        img_reshaped_coeff = img_reshaped_val - img_reshaped_index
        img_reshaped_coeff_one = 1.0 - img_reshaped_coeff

        mapped_color_map_control_y = torch.gather(color_map_control_y, 2, img_reshaped_index)
        mapped_color_map_control_y_plus = torch.gather(color_map_control_y, 2, img_reshaped_index_plus)

        out_img_reshaped = img_reshaped_coeff_one * mapped_color_map_control_y + img_reshaped_coeff * mapped_color_map_control_y_plus
        # for i in range(0, self.control_point):
        #     mask = img_reshaped_index == i
        #     masked_img_reshaped_coeff = mask * img_reshaped_coeff
        #     masked_img_reshaped_coeff_one = mask * img_reshaped_coeff_one
        #     out_img_reshaped += masked_img_reshaped_coeff_one * color_map_control_y[:,:,i:i+1] + masked_img_reshaped_coeff * color_map_control_y[:,:,i+1:i+2]

        out_img_reshaped = out_img_reshaped.reshape(N, C, H, W)
        return out_img_reshaped

class colorTransform5(nn.Module):
    def __init__(self, control_point=16, offset_param=0.04, config=0):
        super(colorTransform5, self).__init__()
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)
        self.control_point = control_point
        self.sigmoid = torch.nn.Sigmoid()
        self.config = config
        self.feature_num = config.feature_num

        self.epsilon = 1e-8
        
        self.offset_param = offset_param
        if self.offset_param != -1:
            if config.trainable_offset == 1:
                self.offset_param = nn.Parameter(torch.tensor([offset_param], dtype=torch.float32))
            else:
                self.offset_param = offset_param

    def forward(self, org_img, params, color_mapping_global_a, color_map_control):
        N, C, H, W = org_img.shape
        #out_img = torch.zeros_like(org_img)
        color_map_control_x = color_map_control.clone()
        if self.offset_param != -1:
            params = params.reshape(N, self.feature_num, self.control_point) * self.offset_param
        else:
            params = params.reshape(N, self.feature_num, self.control_point)
        
        params = torch.cumsum(params, dim=2)
        color_map_control_y = color_map_control_x + params

        color_map_control_y = torch.cat((color_map_control_y, color_map_control_y[:, :, self.control_point-1:self.control_point]), dim=2)
        color_map_control_x = torch.cat((color_map_control_x, color_map_control_x[:, :, self.control_point-1:self.control_point]), dim=2)
        img_reshaped = org_img.reshape(N, self.feature_num, -1)
        #out_img_reshaped = out_img.reshape(N, self.feature_num, -1)
        img_reshaped_val = img_reshaped * (self.control_point-1)


        img_reshaped_index = torch.floor(img_reshaped * (self.control_point-1))
        img_reshaped_index = img_reshaped_index.type(torch.int64)
        img_reshaped_index_plus = img_reshaped_index + 1

        img_reshaped_coeff = img_reshaped_val - img_reshaped_index
        img_reshaped_coeff_one = 1.0 - img_reshaped_coeff

        mapped_color_map_control_y = torch.gather(color_map_control_y, 2, img_reshaped_index)
        mapped_color_map_control_y_plus = torch.gather(color_map_control_y, 2, img_reshaped_index_plus)

        out_img_reshaped = img_reshaped_coeff_one * mapped_color_map_control_y + img_reshaped_coeff * mapped_color_map_control_y_plus
        # for i in range(0, self.control_point):
        #     mask = img_reshaped_index == i
        #     masked_img_reshaped_coeff = mask * img_reshaped_coeff
        #     masked_img_reshaped_coeff_one = mask * img_reshaped_coeff_one
        #     out_img_reshaped += masked_img_reshaped_coeff_one * color_map_control_y[:,:,i:i+1] + masked_img_reshaped_coeff * color_map_control_y[:,:,i+1:i+2]

        out_img_reshaped = out_img_reshaped.reshape(N, C, H, W)
        return out_img_reshaped


class colorTransform6(nn.Module):
    def __init__(self, control_point=16, offset_param=0.04, config=0):
        super(colorTransform6, self).__init__()
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)
        self.control_point = control_point
        self.sigmoid = torch.nn.Sigmoid()
        self.config = config
        self.feature_num = config.feature_num

        self.epsilon = 1e-8

        self.offset_param = offset_param
        if self.offset_param != -1:
            if config.trainable_offset == 1:
                self.offset_param = nn.Parameter(torch.tensor([offset_param], dtype=torch.float32))
            else:
                self.offset_param = offset_param

    def forward(self, org_img, params, color_mapping_global_a, color_map_control):
        N, C, H, W = org_img.shape
        # out_img = torch.zeros_like(org_img)
        color_map_control_x = color_map_control.clone()
        if self.offset_param != -1:
            params = params.reshape(N, self.feature_num, self.control_point) * self.offset_param
        else:
            params = params.reshape(N, self.feature_num, self.control_point)
        color_map_control_y = color_map_control_x + params

        color_map_control_y = torch.cat(
            (color_map_control_y, color_map_control_y[:, :, self.control_point - 1:self.control_point]), dim=2)
        color_map_control_x = torch.cat(
            (color_map_control_x, color_map_control_x[:, :, self.control_point - 1:self.control_point]), dim=2)
        img_reshaped = org_img.reshape(N, self.feature_num, -1)
        # out_img_reshaped = out_img.reshape(N, self.feature_num, -1)
        img_reshaped_val = img_reshaped * (self.control_point - 1)

        img_reshaped_index = torch.floor(img_reshaped * (self.control_point - 1))
        img_reshaped_index = img_reshaped_index.type(torch.int64)
        img_reshaped_index_plus = img_reshaped_index + 1

        img_reshaped_coeff = img_reshaped_val - img_reshaped_index
        img_reshaped_coeff_one = 1.0 - img_reshaped_coeff

        mapped_color_map_control_y = torch.gather(color_map_control_y, 2, img_reshaped_index)
        mapped_color_map_control_y_plus = torch.gather(color_map_control_y, 2, img_reshaped_index_plus)

        out_img_reshaped = img_reshaped_coeff_one * mapped_color_map_control_y + img_reshaped_coeff * mapped_color_map_control_y_plus
        # for i in range(0, self.control_point):
        #     mask = img_reshaped_index == i
        #     masked_img_reshaped_coeff = mask * img_reshaped_coeff
        #     masked_img_reshaped_coeff_one = mask * img_reshaped_coeff_one
        #     out_img_reshaped += masked_img_reshaped_coeff_one * color_map_control_y[:,:,i:i+1] + masked_img_reshaped_coeff * color_map_control_y[:,:,i+1:i+2]

        out_img_reshaped = out_img_reshaped.reshape(N, C, H, W)
        return out_img_reshaped, params

class colorTransform7(nn.Module):
    def __init__(self, control_point=16, offset_param=0.04, config=0):
        super(colorTransform7, self).__init__()
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)
        self.control_point = control_point
        self.sigmoid = torch.nn.Sigmoid()
        self.config = config
        self.feature_num = config.feature_num

        self.epsilon = 1e-8

        self.offset_param = offset_param
        if self.offset_param != -1:
            if config.trainable_offset == 1:
                self.offset_param = nn.Parameter(torch.tensor([offset_param], dtype=torch.float32))
            else:
                self.offset_param = offset_param

    def forward(self, org_img, params, color_mapping_global_a, color_map_control):
        N, C, H, W = org_img.shape
        # out_img = torch.zeros_like(org_img)
        color_map_control_x = color_map_control.clone()
        if self.offset_param != -1:
            params = params.reshape(N, self.feature_num, self.control_point) * self.offset_param
        else:
            params = params.reshape(N, self.feature_num, self.control_point)
        color_map_control_y = color_map_control_x + params

        color_map_control_y = torch.cat(
            (color_map_control_y, color_map_control_y[:, :, self.control_point - 1:self.control_point]), dim=2)
        color_map_control_x = torch.cat(
            (color_map_control_x, color_map_control_x[:, :, self.control_point - 1:self.control_point]), dim=2)
        img_reshaped = org_img.reshape(N, self.feature_num, -1)
        # out_img_reshaped = out_img.reshape(N, self.feature_num, -1)
        img_reshaped_val = img_reshaped * (self.control_point - 1)

        img_reshaped_index = torch.floor(img_reshaped * (self.control_point - 1))
        img_reshaped_index = img_reshaped_index.type(torch.int64)
        img_reshaped_index_plus = img_reshaped_index + 1

        img_reshaped_coeff = img_reshaped_val - img_reshaped_index
        img_reshaped_coeff_one = 1.0 - img_reshaped_coeff

        mapped_color_map_control_y = torch.gather(color_map_control_y, 2, img_reshaped_index)
        mapped_color_map_control_y_plus = torch.gather(color_map_control_y, 2, img_reshaped_index_plus)

        out_img_reshaped = img_reshaped_coeff_one * mapped_color_map_control_y + img_reshaped_coeff * mapped_color_map_control_y_plus
        # for i in range(0, self.control_point):
        #     mask = img_reshaped_index == i
        #     masked_img_reshaped_coeff = mask * img_reshaped_coeff
        #     masked_img_reshaped_coeff_one = mask * img_reshaped_coeff_one
        #     out_img_reshaped += masked_img_reshaped_coeff_one * color_map_control_y[:,:,i:i+1] + masked_img_reshaped_coeff * color_map_control_y[:,:,i+1:i+2]

        out_img_reshaped = out_img_reshaped.reshape(N, C, H, W)
        return out_img_reshaped, color_map_control_y
class colorTransform_multi(nn.Module):
    def __init__(self, control_point=16, offset_param=0.04, num_weight=1, config=0):
        super(colorTransform_multi, self).__init__()
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)
        self.control_point = control_point
        self.sigmoid = torch.nn.Sigmoid()
        self.config = config
        self.feature_num = config.feature_num
        self.num_weight = num_weight
        self.epsilon = 1e-8

        self.offset_param = nn.Parameter(torch.tensor([offset_param], dtype=torch.float32))



    def forward(self, org_img, params, color_mapping_global_a, color_map_control, weight_map):
        N, C, H, W = org_img.shape
        #out_img = torch.zeros_like(org_img)

        

        img_reshaped = org_img.reshape(N, self.feature_num, -1)
        weight_map = weight_map.reshape(N,self.num_weight, -1)
        out_img_reshaped = torch.zeros_like(img_reshaped)

        img_reshaped_val = img_reshaped * (self.control_point-1)

        img_reshaped_index = torch.floor(img_reshaped * (self.control_point-1))
        img_reshaped_index = img_reshaped_index.type(torch.int64)
        img_reshaped_index_plus = img_reshaped_index + 1

        img_reshaped_coeff = img_reshaped_val - img_reshaped_index
        img_reshaped_coeff_one = 1.0 - img_reshaped_coeff
        
        params = params.reshape(N, self.feature_num, self.control_point, self.num_weight) * self.offset_param
        
        for w in range(0, self.num_weight):
            cur_params = params[:,:,:, w] * self.offset_param
            cur_weight_map = weight_map[:,w:w+1,:]

            color_map_control_x = color_map_control.clone()
        
            color_map_control_y = color_map_control_x + cur_params

            color_map_control_x = torch.cat((color_map_control_x, color_map_control_x[:, :, self.control_point-1:self.control_point]), dim=2)
            color_map_control_y = torch.cat((color_map_control_y, color_map_control_y[:, :, self.control_point-1:self.control_point]), dim=2)
            
            
            #out_img_reshaped = out_img.reshape(N, self.feature_num, -1)
 
            mapped_color_map_control_y = torch.gather(color_map_control_y, 2, img_reshaped_index)
            mapped_color_map_control_y_plus = torch.gather(color_map_control_y, 2, img_reshaped_index_plus)

            out_img_reshaped += cur_weight_map * (img_reshaped_coeff_one * mapped_color_map_control_y + img_reshaped_coeff * mapped_color_map_control_y_plus)


        # for i in range(0, self.control_point):
        #     mask = img_reshaped_index == i
        #     masked_img_reshaped_coeff = mask * img_reshaped_coeff
        #     masked_img_reshaped_coeff_one = mask * img_reshaped_coeff_one
        #     out_img_reshaped += masked_img_reshaped_coeff_one * color_map_control_y[:,:,i:i+1] + masked_img_reshaped_coeff * color_map_control_y[:,:,i+1:i+2]

        out_img_reshaped = out_img_reshaped.reshape(N, C, H, W)
        return out_img_reshaped
    
class colorTransform_xoffset(nn.Module):
    def __init__(self, control_point=16, offset_param=0.04, offset_param2=0.1, config=0):
        super(colorTransform_xoffset, self).__init__()
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)
        self.control_point = control_point
        self.sigmoid = torch.nn.Sigmoid()
        self.config = config
        self.feature_num = config.feature_num

        self.epsilon = 1e-8

        self.offset_param = nn.Parameter(torch.tensor([offset_param], dtype=torch.float32))
        if offset_param2 == 0:
            self.offset_param2 = self.offset_param
        else:
            self.offset_param2 = nn.Parameter(torch.tensor([offset_param2], dtype=torch.float32))


    def forward(self, org_img, params, color_mapping_global_a, color_map_control):
        N, C, H, W = org_img.shape
        #out_img = torch.zeros_like(org_img)
        color_map_control_x = color_map_control.clone()

        y_params = params[:,:self.control_point*self.feature_num]
        x_params = params[:,self.control_point*self.feature_num:]

        y_params = y_params.reshape(N, self.feature_num, -1) * self.offset_param
        x_params = x_params.reshape(N, self.feature_num, -1) * self.offset_param2
        #params = params.reshape(N, self.feature_num, -1) * self.offset_param

        temp1 = torch.zeros(N, self.feature_num, 1).cuda(self.config.rank)
        temp2 = torch.zeros(N, self.feature_num, 1).cuda(self.config.rank)

        img_reshaped_index = -1 *torch.ones(N, C, H * W).cuda(self.config.rank)

        #y_params = params[:,:,:self.control_point]
        x_params = torch.cat((temp1, x_params, temp2), dim=2)

        color_map_control_y = color_map_control_x + y_params
        color_map_control_x = color_map_control_x + x_params

        color_map_control_y = torch.cat((color_map_control_y, color_map_control_y[:, :, self.control_point-1:self.control_point]), dim=2)
        color_map_control_x = torch.cat((color_map_control_x, color_map_control_x[:, :, self.control_point-1:self.control_point]), dim=2)

        color_map_control_x, sort_x_idx = torch.sort(color_map_control_x, dim=2)
        color_map_control_y = torch.gather(input=color_map_control_y, dim=2, index=sort_x_idx)

        img_reshaped = org_img.reshape(N, self.feature_num, -1)
        
        #img_reshaped_index_plus = torch.sum((img_reshaped.unsqueeze(2).repeat(1,1,self.control_point,1) - color_map_control_x[:,:,:-1].unsqueeze(3)) >= 0,dim=2).type(torch.int64)
        #img_reshaped_index = img_reshaped_index_plus - 1
        for i in range(0,self.control_point):
            temp = (img_reshaped - color_map_control_x[:,:,i:i+1]) >= 0
            img_reshaped_index += temp

        img_reshaped_index = img_reshaped_index.type(torch.int64)
        img_reshaped_index_plus = img_reshaped_index + 1

        mapped_color_map_control_x = torch.gather(color_map_control_x, 2, img_reshaped_index)
        mapped_color_map_control_x_plus = torch.gather(color_map_control_x, 2, img_reshaped_index_plus)

        mapped_color_map_control_y = torch.gather(color_map_control_y, 2, img_reshaped_index)
        mapped_color_map_control_y_plus = torch.gather(color_map_control_y, 2, img_reshaped_index_plus)

        img_reshaped_coeff = (img_reshaped - mapped_color_map_control_x) / (mapped_color_map_control_x_plus - mapped_color_map_control_x + 1e-10)

        out_img_reshaped =  (1-img_reshaped_coeff) * mapped_color_map_control_y + img_reshaped_coeff * mapped_color_map_control_y_plus

        #img_reshaped_val = img_reshaped * (self.control_point-1)

        #img_reshaped_index = torch.floor(img_reshaped * (self.control_point-1))
        #img_reshaped_index = img_reshaped_index.type(torch.int64)
        #img_reshaped_index_plus = img_reshaped_index + 1

        #img_reshaped_coeff = img_reshaped_val - img_reshaped_index
        #img_reshaped_coeff_one = 1.0 - img_reshaped_coeff

        #mapped_color_map_control_y = torch.gather(color_map_control_y, 2, img_reshaped_index)
        #mapped_color_map_control_y_plus = torch.gather(color_map_control_y, 2, img_reshaped_index_plus)

        #out_img_reshaped = img_reshaped_coeff_one * mapped_color_map_control_y + img_reshaped_coeff * mapped_color_map_control_y_plus

        out_img_reshaped = out_img_reshaped.reshape(N, C, H, W)
        return out_img_reshaped


class colorTransform_xoffset_softmax(nn.Module):
    def __init__(self, control_point=16, offset_param=0.04, offset_param2=0.04, config=0):
        super(colorTransform_xoffset_softmax, self).__init__()
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)
        self.control_point = control_point
        self.sigmoid = torch.nn.Sigmoid()
        self.config = config
        self.feature_num = config.feature_num

        self.epsilon = 1e-8

        self.offset_param = nn.Parameter(torch.tensor([offset_param], dtype=torch.float32))
        if offset_param2 == 0:
            self.offset_param2 = self.offset_param
        else:
            self.offset_param2 = nn.Parameter(torch.tensor([offset_param2], dtype=torch.float32))

    def forward(self, org_img, params, color_mapping_global_a, color_map_control):
        N, C, H, W = org_img.shape
        # out_img = torch.zeros_like(org_img)
        color_map_control_x = color_map_control.clone()

        y_params = params[:, :self.feature_num * self.control_point]
        x_params = params[:,  self.feature_num * self.control_point:]

        y_params = y_params.reshape(N, self.feature_num, -1) * self.offset_param
        x_params = x_params.reshape(N, self.feature_num, -1) * self.offset_param2

        temp1 = torch.zeros(N, self.feature_num, 1).cuda(self.config.rank)


        img_reshaped_index = -1 * torch.ones(N, C, H * W).cuda(self.config.rank)


        m = nn.Softmax(dim=2)
        x_params = m(x_params)
        x_params = torch.cumsum(x_params, dim=2)

        x_params = torch.cat((temp1, x_params), dim=2)

        color_map_control_y = x_params + y_params
        color_map_control_x = x_params

        color_map_control_y = torch.cat((color_map_control_y, color_map_control_y[:, :, self.control_point - 1:self.control_point]), dim=2)
        color_map_control_x = torch.cat(
            (color_map_control_x, color_map_control_x[:, :, self.control_point - 1:self.control_point]), dim=2)

        #color_map_control_x, sort_x_idx = torch.sort(color_map_control_x, dim=2)
        #color_map_control_y = torch.gather(input=color_map_control_y, dim=2, index=sort_x_idx)

        img_reshaped = org_img.reshape(N, self.feature_num, -1)

        # img_reshaped_index_plus = torch.sum((img_reshaped.unsqueeze(2).repeat(1,1,self.control_point,1) - color_map_control_x[:,:,:-1].unsqueeze(3)) >= 0,dim=2).type(torch.int64)
        # img_reshaped_index = img_reshaped_index_plus - 1
        for i in range(0, self.control_point):
            temp = (img_reshaped - color_map_control_x[:, :, i:i + 1]) >= 0
            img_reshaped_index += temp

        img_reshaped_index = img_reshaped_index.type(torch.int64)
        img_reshaped_index_plus = img_reshaped_index + 1

        mapped_color_map_control_x = torch.gather(color_map_control_x, 2, img_reshaped_index)
        mapped_color_map_control_x_plus = torch.gather(color_map_control_x, 2, img_reshaped_index_plus)

        mapped_color_map_control_y = torch.gather(color_map_control_y, 2, img_reshaped_index)
        mapped_color_map_control_y_plus = torch.gather(color_map_control_y, 2, img_reshaped_index_plus)

        img_reshaped_coeff = (img_reshaped - mapped_color_map_control_x) / (
                    mapped_color_map_control_x_plus - mapped_color_map_control_x + 1e-10)

        out_img_reshaped = (1 - img_reshaped_coeff) * mapped_color_map_control_y + img_reshaped_coeff * mapped_color_map_control_y_plus

        # img_reshaped_val = img_reshaped * (self.control_point-1)

        # img_reshaped_index = torch.floor(img_reshaped * (self.control_point-1))
        # img_reshaped_index = img_reshaped_index.type(torch.int64)
        # img_reshaped_index_plus = img_reshaped_index + 1

        # img_reshaped_coeff = img_reshaped_val - img_reshaped_index
        # img_reshaped_coeff_one = 1.0 - img_reshaped_coeff

        # mapped_color_map_control_y = torch.gather(color_map_control_y, 2, img_reshaped_index)
        # mapped_color_map_control_y_plus = torch.gather(color_map_control_y, 2, img_reshaped_index_plus)

        # out_img_reshaped = img_reshaped_coeff_one * mapped_color_map_control_y + img_reshaped_coeff * mapped_color_map_control_y_plus

        out_img_reshaped = out_img_reshaped.reshape(N, C, H, W)
        return out_img_reshaped
def initialize_weights_part(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
class resnet18_224(nn.Module):

    def __init__(self, out_dim=5, res_num=18, res_size=224, aug_test=False, fc_num=1, init_w=0):
        super(resnet18_224, self).__init__()

        self.aug_test = aug_test
        if res_num == 18:
            net = models.resnet18(pretrained=True)
        elif res_num == 34:
            net = models.resnet34(pretrained=True)
        elif res_num == 50:
            net = models.resnet50(pretrained=True)
        elif res_num == 101:
            net = models.resnet101(pretrained=True)
        elif res_num == 16:
            net = models.vgg16(pretrained=True)
        elif res_num == 19:
            net = models.vgg19(pretrained=True)
        if res_size == 0:
            self.upsample = nn.Identity()
        else:
            self.upsample = nn.Upsample(size=(res_size, res_size), mode='bilinear')
        if res_num == 50 or res_num == 101:
            net.fc = nn.Linear(2048, out_dim)
            if init_w > 0:
                initialize_weights_part(net.fc)

        elif res_num == 18 or res_num == 34:
            if fc_num == 1:
                net.fc = nn.Linear(512, out_dim)
            elif fc_num == 2:
                lists = []
                lists += [nn.Linear(512, 1024),
                          #nn.BatchNorm2d(1024),
                          nn.ReLU(),
                          nn.Linear(1024, out_dim)]
                net.fc = nn.Sequential(*lists)
            elif fc_num == 3:    
                lists = []
                lists += [nn.Linear(512, 1024),
                          #nn.BatchNorm2d(1024),
                          nn.ReLU(),
                          nn.Linear(1024, 2048),
                          #nn.BatchNorm2d(2048),
                          nn.ReLU(),
                          nn.Linear(2048, out_dim)]
                net.fc = nn.Sequential(*lists)
            if init_w > 0:
                initialize_weights_part(net.fc)
            
            #multi layer...
        elif res_num == 16 or res_num == 19:
            net.classifier[-1] = nn.Linear(4096, out_dim)
            if init_w > 0:
                nn.init.kaiming_normal_(net.classifier[-1].weight)
                if net.classifier[-1].bias is not None:
                    nn.init.constant_(net.classifier[-1].bias, 0.0)
        self.model = net


    def forward(self, x):
        x = self.upsample(x)
        if self.aug_test:
            # x = torch.cat((x, torch.rot90(x, 1, [2, 3]), torch.rot90(x, 3, [2, 3])), 0)
            x = torch.cat((x, torch.flip(x, [3])), 0)
        f = self.model(x)
        return f


class resnet18_224_2(nn.Module):

    def __init__(self, out_dim=5, out_dim2=0, out_dim3=0, out_dim4=0, res_num=18, res_size=224, aug_test=False, fc_num=2, init_w=0, init_w2=0, init_w_last=0, fc_node=1024, fc_node1=1024, fc_node2=1024):
        super(resnet18_224_2, self).__init__()

        self.aug_test = aug_test
        self.out_dim2 = out_dim2
        self.out_dim3 = out_dim3
        self.out_dim4 = out_dim4
        self.fc_num = fc_num
        if res_num == 5:
            net = TPAMIBackbone(input_resolution=res_size)
        if res_num == 18:
            net = models.resnet18(pretrained=True)
        elif res_num == 34:
            net = models.resnet34(pretrained=True)
        elif res_num == 50:
            net = models.resnet50(pretrained=True)
        elif res_num == 101:
            net = models.resnet101(pretrained=True)
        elif res_num == 16:
            net = models.vgg16(pretrained=True)
        elif res_num == 19:
            net = models.vgg19(pretrained=True)
        if res_size == 0:
            self.upsample = nn.Identity()
        else:
            self.upsample = nn.Upsample(size=(res_size, res_size), mode='bilinear')
        if res_num == 50 or res_num == 101:
            net.fc = nn.Identity()
            lists = []
            lists += [nn.Linear(2048, fc_node1),
                      # nn.BatchNorm2d(1024),
                      nn.ReLU(),
                      nn.Linear(fc_node1, out_dim)]
            self.fc = nn.Sequential(*lists)
            if init_w == -1:
                torch.nn.init.constant_(self.fc[2].weight.data, 0)
                torch.nn.init.constant_(self.fc[2].bias.data, 0)
            if init_w == 0:
                initialize_weights_part(self.fc)
                torch.nn.init.constant_(self.fc[2].weight.data, 0)
                torch.nn.init.constant_(self.fc[2].bias.data, 0)

            if out_dim2 > 0:
                lists = []
                lists += [nn.Linear(2048, fc_node2),
                          # nn.BatchNorm2d(1024),
                          nn.ReLU(),
                          nn.Linear(fc_node2, out_dim2)]
                self.fc2 = nn.Sequential(*lists)
                if init_w2 == 3:
                    initialize_weights_part(self.fc2)
                if init_w2 == 4:
                    initialize_weights_part(self.fc2[0])
                    torch.nn.init.constant_(self.fc2[2].weight.data, 0)
                    torch.nn.init.constant_(self.fc2[2].bias.data, 0)
                    
            if out_dim3 > 0:
                self.fc3 = nn.Linear(2048, out_dim3)
                if init_w2 == 1:
                    torch.nn.init.constant_(self.fc3.weight.data, 0)
                    torch.nn.init.constant_(self.fc3.bias.data, 1)
                if init_w2 == 2:
                    torch.nn.init.constant_(self.fc3.weight.data, 0)
                    torch.nn.init.constant_(self.fc3.bias.data, 0)
            if out_dim4 > 0:
                lists = []
                lists += [nn.Linear(2048, fc_node2),
                          # nn.BatchNorm2d(1024),
                          nn.ReLU(),
                          nn.Linear(fc_node2, out_dim4)]
                self.fc4 = nn.Sequential(*lists)
                if init_w_last == 0:
                    initialize_weights_part(self.fc4)
                elif init_w_last == 1:
                    torch.nn.init.constant_(self.fc4[2].weight.data, 0)
                    torch.nn.init.constant_(self.fc4[2].bias.data, 1.0/64.0)
                elif init_w_last == 2:
                    initialize_weights_part(self.fc4)
                    torch.nn.init.constant_(self.fc4[2].weight.data, 0)
                    torch.nn.init.constant_(self.fc4[2].bias.data, 1.0/64.0)
                elif init_w_last == 3:
                    initialize_weights_part(self.fc4)
                    torch.nn.init.constant_(self.fc4[2].weight.data, 0)
                    torch.nn.init.constant_(self.fc4[2].bias.data, 1.0)

            if init_w > 0:
                initialize_weights_part(net.fc)

        elif res_num == 18 or res_num == 34 or res_num == 5:
            net.fc = nn.Identity()
            # 1
            if self.fc_num != 1:
                lists = []
                lists += [nn.Linear(512, fc_node1),
                        # nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Linear(fc_node1, out_dim)]
                self.fc = nn.Sequential(*lists)
                if init_w == -1:
                    torch.nn.init.constant_(self.fc[2].weight.data, 0)
                    torch.nn.init.constant_(self.fc[2].bias.data, 0)
                if init_w == 0:
                    initialize_weights_part(self.fc)
                    torch.nn.init.constant_(self.fc[2].weight.data, 0)
                    torch.nn.init.constant_(self.fc[2].bias.data, 0)

                # 2
                if out_dim2 > 0:
                    lists = []
                    lists += [nn.Linear(512, fc_node2),
                            # nn.BatchNorm2d(1024),
                            nn.ReLU(),
                            nn.Linear(fc_node2, out_dim2)]
                    self.fc2 = nn.Sequential(*lists)
                    if init_w2 == 3:
                        initialize_weights_part(self.fc2)
                    if init_w2 == 4:
                        initialize_weights_part(self.fc2[0])
                        torch.nn.init.constant_(self.fc2[2].weight.data, 0)
                        torch.nn.init.constant_(self.fc2[2].bias.data, 0)
                # 3 (ignore)
                if out_dim3 > 0:
                    self.fc3 = nn.Linear(512, out_dim3)
                    if init_w2 == 1:
                        torch.nn.init.constant_(self.fc3.weight.data, 0)
                        torch.nn.init.constant_(self.fc3.bias.data, 1)
                    if init_w2 == 2:
                        torch.nn.init.constant_(self.fc3.weight.data, 0)
                        torch.nn.init.constant_(self.fc3.bias.data, 0)
                # 4
                if out_dim4 > 0:
                    lists = []
                    lists += [nn.Linear(512, fc_node2),
                            # nn.BatchNorm2d(1024),
                            nn.ReLU(),
                            nn.Linear(fc_node2, out_dim4)]
                    self.fc4 = nn.Sequential(*lists)
                    if init_w_last == 0:
                        initialize_weights_part(self.fc4)
                    elif init_w_last == 1:
                        torch.nn.init.constant_(self.fc4[2].weight.data, 0)
                        torch.nn.init.constant_(self.fc4[2].bias.data, 1.0 / 64.0)
                    elif init_w_last == 2:
                        initialize_weights_part(self.fc4)
                        torch.nn.init.constant_(self.fc4[2].weight.data, 0)
                        torch.nn.init.constant_(self.fc4[2].bias.data, 1.0/64.0)
                    elif init_w_last == 3:
                        initialize_weights_part(self.fc4)
                        torch.nn.init.constant_(self.fc4[2].weight.data, 0)
                        torch.nn.init.constant_(self.fc4[2].bias.data, 1.0)

                if init_w > 0:
                    initialize_weights_part(net.fc)
            elif self.fc_num == 1:
                lists = []
                lists += [nn.Linear(512, out_dim)]
                self.fc = nn.Sequential(*lists)
                if init_w == -1:
                    torch.nn.init.constant_(self.fc[0].weight.data, 0)
                    torch.nn.init.constant_(self.fc[0].bias.data, 0)

                # 2
                if out_dim2 > 0:
                    lists = []
                    lists += [nn.Linear(512, out_dim2)]
                    self.fc2 = nn.Sequential(*lists)
                # 3 (ignore)
                if out_dim3 > 0:
                    self.fc3 = nn.Linear(512, out_dim3)
                    if init_w2 == 1:
                        torch.nn.init.constant_(self.fc3.weight.data, 0)
                        torch.nn.init.constant_(self.fc3.bias.data, 1)
                    if init_w2 == 2:
                        torch.nn.init.constant_(self.fc3.weight.data, 0)
                        torch.nn.init.constant_(self.fc3.bias.data, 0)
                # 4
                if out_dim4 > 0:
                    lists = []
                    lists += [nn.Linear(512,  out_dim4)]
                    self.fc4 = nn.Sequential(*lists)
                    if init_w_last == 0:
                        initialize_weights_part(self.fc4)
                    elif init_w_last == 1:
                        torch.nn.init.constant_(self.fc4[0].weight.data, 0)
                        torch.nn.init.constant_(self.fc4[0].bias.data, 1.0 / 64.0)

                if init_w > 0:
                    initialize_weights_part(net.fc)
                

            # multi layer...
        elif res_num == 16 or res_num == 19:
            net.classifier[-1] = nn.Linear(4096, out_dim)
            if init_w > 0:
                nn.init.kaiming_normal_(net.classifier[-1].weight)
                if net.classifier[-1].bias is not None:
                    nn.init.constant_(net.classifier[-1].bias, 0.0)
        self.model = net

    def forward(self, x):
        x = self.upsample(x)
        if self.aug_test:
            # x = torch.cat((x, torch.rot90(x, 1, [2, 3]), torch.rot90(x, 3, [2, 3])), 0)
            x = torch.cat((x, torch.flip(x, [3])), 0)
        f = self.model(x)
        f1 = self.fc(f)
        if self.out_dim2 > 0:
            f2 = self.fc2(f)
            f1 = torch.cat((f1, f2), dim=1)
        if self.out_dim3 > 0:
            f3 = self.fc3(f)
            f1 = torch.cat((f1, f3), dim=1)
        if self.out_dim4 > 0:
            f4 = self.fc4(f)
            f1 = torch.cat((f1, f4), dim=1)
        return f1