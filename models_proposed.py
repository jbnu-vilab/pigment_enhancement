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
    def __init__(self, input_feature, output_feature, ksize=3, stride=2, pad=1, extra_conv=False, act='silu'):
        super(convBlock2, self).__init__()
        if act=='silu':
            activation = nn.SiLU(inplace=True)
        elif act=='gelu':
            activation = nn.GELU()

        lists = []
        if extra_conv:
            lists += [
                nn.Conv2d(input_feature, output_feature, kernel_size=(ksize, ksize), stride=(stride, stride), padding=(pad, pad)),
                nn.BatchNorm2d(output_feature),
                activation,  # Swish activation
                nn.Conv2d(output_feature, output_feature, kernel_size=(ksize, ksize), stride=(stride, stride), padding=(pad, pad))
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


        temp1 = torch.zeros(N, 3, 1).cuda()
        temp2 = torch.ones(N, 3, 1).cuda()
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


class DCPNet(nn.Module):
    def __init__(self, config):
        super(DCPNet, self).__init__()

        self.scale = config.scale
        self.act = config.act

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)
        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        self.transform = T.Resize((256, 256))
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point

        self.feature_num = self.control_point * 3 * 3 + 15

        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128, output_feature=self.feature_num, ksize=1, stride=1,
                                                pad=0, extra_conv=True, act=self.act)


        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m


        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma, config.trainable_offset, config.offset_param, config.offset_param2, config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                    mlp_dim=512,
                                    pool='cls', dim_head=64, dropout=0.1)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=256, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)

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
        img = self.transform(org_img)

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.scale == 4:
            x4 = self.conv4(x3)
            x6 = self.transformer(x4)
        elif self.scale == 5:
            x4 = self.conv4(x3)
            x5 = self.conv5(x4)
            x6 = self.transformer(x5)


        if self.scale == 4:
            N, C, H, W = x4.shape
        elif self.scale == 5:
            N, C, H, W = x5.shape

        x6 = torch.transpose(x6, 1, 2)
        x6_global = x6[:, :, 0]
        x6_global = x6_global.unsqueeze(2).unsqueeze(3)
        x6_local = x6[:, :, 1:]
        x6_local = x6_local.reshape(N, C, H, W)
        # not used
        color_mapping_local = self.color_conv_local(x6_local)

        color_mapping_global = self.color_conv_global(x6_global)
        N, C, H, W = color_mapping_global.shape
        color_mapping_global = color_mapping_global.reshape(N, C)
        if self.global_module == 1:
            y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        else:
            y = org_img
        if self.residual == 1:
            y1 = self.res1(y)
            y2 = self.res2(y1)
            y3 = self.res3(y2)
            y3 = self.res4(y3)
            y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
            y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
            y1 = self.res7(y1)
            y_tot = y + y1
        else:
            y_tot = y

        return y_tot





class DCPNet2(nn.Module):
    def __init__(self, config):
        super(DCPNet2, self).__init__()

        self.scale = config.scale
        self.act = config.act

        self.conv_emb = convBlock(input_feature=3, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)
        self.conv_out = convBlock(input_feature=64, output_feature=3, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)

        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)


        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        self.transform = T.Resize((256, 256))
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode

        #self.feature_num = self.control_point * 3 * 3 + 15
        self.feature_num = 128
        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128, output_feature=self.feature_num, ksize=1, stride=1,
                                                pad=0, extra_conv=True, act=self.act)


        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m


        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma, config.trainable_offset, config.offset_param, config.offset_param2, config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                    mlp_dim=512,
                                    pool='cls', dim_head=64, dropout=0.1)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=256, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)

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
        img = self.transform(org_img)

        img_f = self.conv_emb(org_img)

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.scale == 4:
            x4 = self.conv4(x3)
            x6 = self.transformer(x4)
        #elif self.scale == 5:
        #    x4 = self.conv4(x3)
        #    x5 = self.conv5(x4)
        #    x6 = self.transformer(x5)


        if self.scale == 4:
            N, C, H, W = x4.shape
        #elif self.scale == 5:
        #    N, C, H, W = x5.shape

        x6 = torch.transpose(x6, 1, 2)
        x6_global = x6[:, :, 0]
        x6_global = x6_global.unsqueeze(2).unsqueeze(3)
        x6_local = x6[:, :, 1:]
        x6_local = x6_local.reshape(N, C, H, W)

        color_mapping_global = self.color_conv_global(x6_global)
        gamma_g = color_mapping_global[:, :64, ]
        beta_g = color_mapping_global[:, 64:, ]



        x4 = self.conv4_1(x4)
        x3 = self.conv3_1(torch.cat((x3, F. upsample_bilinear(x4, scale_factor=2)), dim=1))
        x2 = self.conv2_1(torch.cat((x2, F. upsample_bilinear(x3, scale_factor=2)), dim=1))
        x1 = self.conv1_1(torch.cat((x1, F.upsample_bilinear(x2, scale_factor=2)), dim=1))

        _, _, ho, wo = org_img.shape
        resize = T.Resize((ho,wo))

        # x1 to original size
        x1 = resize(x1)
        gamma_l = x1[:, :64, :, :]
        beta_l = x1[:, 64:, :, :]

        # Get global transform..
        if self.glo_mode == 0:
            y = gamma_g * img_f + beta_g

        elif self.glo_mode == 1:
            # global + local
            y = (gamma_g + gamma_l) * img_f + (beta_g + beta_l)
        elif self.glo_mode == 2:
            # global , local separately
            y = gamma_g * img_f + beta_g
            y = gamma_l * y + beta_l
        elif self.glo_mode == 3:
            # local only
            y = gamma_l * img_f + beta_l

        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)


        # self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
        #                          extra_conv=True)

        N, C, H, W = color_mapping_global.shape

        y_tot = self.conv_out(y)

        # # color_mapping_global = color_mapping_global.reshape(N, C)
        # # if self.global_module == 1:
        # #     y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        # # else:
        # #     y = org_img
        # if self.residual == 1:
        #     y1 = self.res1(y)
        #     y2 = self.res2(y1)
        #     y3 = self.res3(y2)
        #     y3 = self.res4(y3)
        #     y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)
        #     y_tot = y + y1
        # else:
        #     y_tot = y
        # # Get local transform
        # # y = rx + b

        return y_tot

class DCPNet3(nn.Module):
    def __init__(self, config):
        super(DCPNet3, self).__init__()

        self.scale = config.scale
        self.act = config.act

        self.conv_emb = convBlock(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)
        self.conv_out = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)

        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)


        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        self.transform = T.Resize((256, 256))
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode

        #self.feature_num = self.control_point * 3 * 3 + 15
        self.feature_num = 128
        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128, output_feature=self.feature_num, ksize=1, stride=1,
                                                pad=0, extra_conv=True, act=self.act)


        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m


        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma, config.trainable_offset, config.offset_param, config.offset_param2, config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                    mlp_dim=512,
                                    pool='cls', dim_head=64, dropout=0.1)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=256, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)

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
        img = self.transform(org_img)

        img_f = self.conv_emb(org_img)

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.scale == 4:
            x4 = self.conv4(x3)
            x6 = self.transformer(x4)
        #elif self.scale == 5:
        #    x4 = self.conv4(x3)
        #    x5 = self.conv5(x4)
        #    x6 = self.transformer(x5)


        if self.scale == 4:
            N, C, H, W = x4.shape
        #elif self.scale == 5:
        #    N, C, H, W = x5.shape

        x6 = torch.transpose(x6, 1, 2)
        x6_global = x6[:, :, 0]
        x6_global = x6_global.unsqueeze(2).unsqueeze(3)
        x6_local = x6[:, :, 1:]
        x6_local = x6_local.reshape(N, C, H, W)

        color_mapping_global = self.color_conv_global(x6_global)
        gamma_g = color_mapping_global[:, :16, ]
        beta_g = color_mapping_global[:, 16:32, ]



        x4 = self.conv4_1(x4)
        x3 = self.conv3_1(torch.cat((x3, F. upsample_bilinear(x4, scale_factor=2)), dim=1))
        x2 = self.conv2_1(torch.cat((x2, F. upsample_bilinear(x3, scale_factor=2)), dim=1))
        x1 = self.conv1_1(torch.cat((x1, F.upsample_bilinear(x2, scale_factor=2)), dim=1))

        _, _, ho, wo = org_img.shape
        resize = T.Resize((ho,wo))

        # x1 to original size
        x1 = resize(x1)
        gamma_l = x1[:, :64, :, :]
        beta_l = x1[:, 64:, :, :]

        # Get global transform..
        if self.glo_mode == 0:
            y = gamma_g * img_f + beta_g

        elif self.glo_mode == 1:
            # global + local
            y = (gamma_g + gamma_l) * img_f + (beta_g + beta_l)
        elif self.glo_mode == 2:
            # global , local separately
            y = gamma_g * img_f + beta_g
            y = gamma_l * y + beta_l
        elif self.glo_mode == 3:
            # local only
            y = gamma_l * img_f + beta_l

        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)


        # self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
        #                          extra_conv=True)

        N, C, H, W = color_mapping_global.shape

        y_tot = self.conv_out(y)

        # # color_mapping_global = color_mapping_global.reshape(N, C)
        # # if self.global_module == 1:
        # #     y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        # # else:
        # #     y = org_img
        # if self.residual == 1:
        #     y1 = self.res1(y)
        #     y2 = self.res2(y1)
        #     y3 = self.res3(y2)
        #     y3 = self.res4(y3)
        #     y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)
        #     y_tot = y + y1
        # else:
        #     y_tot = y
        # # Get local transform
        # # y = rx + b

        return y_tot


class DCPNet4(nn.Module):
    def __init__(self, config):
        super(DCPNet4, self).__init__()

        self.scale = config.scale
        self.act = config.act

        self.conv_emb = convBlock(input_feature=3, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)
        self.conv_out = convBlock(input_feature=64, output_feature=3, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)

        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)


        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        self.transform = T.Resize((256, 256))
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode

        #self.feature_num = self.control_point * 3 * 3 + 15
        self.feature_num = 128
        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128, output_feature=256, ksize=1, stride=1,
                                                pad=0, extra_conv=True, act=self.act)


        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m


        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma, config.trainable_offset, config.offset_param, config.offset_param2, config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                    mlp_dim=512,
                                    pool='cls', dim_head=64, dropout=0.1)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=256, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)

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
        img = self.transform(org_img)

        img_f = self.conv_emb(org_img)

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.scale == 4:
            x4 = self.conv4(x3)
            x6 = self.transformer(x4)
        #elif self.scale == 5:
        #    x4 = self.conv4(x3)
        #    x5 = self.conv5(x4)
        #    x6 = self.transformer(x5)


        if self.scale == 4:
            N, C, H, W = x4.shape
        #elif self.scale == 5:
        #    N, C, H, W = x5.shape

        x6 = torch.transpose(x6, 1, 2)
        x6_global = x6[:, :, 0]
        x6_global = x6_global.unsqueeze(2).unsqueeze(3)
        x6_local = x6[:, :, 1:]
        x6_local = x6_local.reshape(N, C, H, W)

        color_mapping_global = self.color_conv_global(x6_global)
        gamma_g1 = color_mapping_global[:, :64, ]
        beta_g1 = color_mapping_global[:, 64:128, ]
        gamma_g2 = color_mapping_global[:, 128:192, ]
        beta_g2 = color_mapping_global[:, 192:, ]



        x4 = self.conv4_1(x4)
        x3 = self.conv3_1(torch.cat((x3, F. upsample_bilinear(x4, scale_factor=2)), dim=1))
        x2 = self.conv2_1(torch.cat((x2, F. upsample_bilinear(x3, scale_factor=2)), dim=1))
        x1 = self.conv1_1(torch.cat((x1, F.upsample_bilinear(x2, scale_factor=2)), dim=1))

        _, _, ho, wo = org_img.shape
        resize = T.Resize((ho,wo))

        # x1 to original size
        x1 = resize(x1)
        gamma_l = x1[:, :64, :, :]
        beta_l = x1[:, 64:, :, :]

        # Get global transform..
        if self.glo_mode == 0:
            y = gamma_g1 * img_f + beta_g1
            y = gamma_g2 * y + beta_g2

        elif self.glo_mode == 1:
            # global + local
            y = (gamma_g + gamma_l) * img_f + (beta_g + beta_l)
        elif self.glo_mode == 2:
            # global , local separately
            y = gamma_g * img_f + beta_g
            y = gamma_l * y + beta_l
        elif self.glo_mode == 3:
            # local only
            y = gamma_l * img_f + beta_l

        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)


        # self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
        #                          extra_conv=True)

        N, C, H, W = color_mapping_global.shape

        y_tot = self.conv_out(y)
        if self.residual == 1:
            y1 = self.res1(y_tot)
            y2 = self.res2(y1)
            y3 = self.res3(y2)
            y3 = self.res4(y3)
            y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
            y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
            y1 = self.res7(y1)
            y_tot = y_tot + y1

        # # color_mapping_global = color_mapping_global.reshape(N, C)
        # # if self.global_module == 1:
        # #     y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        # # else:
        # #     y = org_img
        # if self.residual == 1:
        #     y1 = self.res1(y)
        #     y2 = self.res2(y1)
        #     y3 = self.res3(y2)
        #     y3 = self.res4(y3)
        #     y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)
        #     y_tot = y + y1
        # else:
        #     y_tot = y
        # # Get local transform
        # # y = rx + b

        return y_tot
    
    
class DCPNet5(nn.Module):
    def __init__(self, config):
        super(DCPNet5, self).__init__()

        self.scale = config.scale
        self.act = config.act

        self.conv_emb = convBlock(input_feature=3, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)
        self.conv_out = convBlock(input_feature=64, output_feature=3, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)

        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)


        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        self.transform = T.Resize((256, 256))
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode

        #self.feature_num = self.control_point * 3 * 3 + 15
        self.feature_num = 128
        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128, output_feature=1024, ksize=1, stride=1,
                                                pad=0, extra_conv=True, act=self.act)


        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m


        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma, config.trainable_offset, config.offset_param, config.offset_param2, config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                    mlp_dim=512,
                                    pool='cls', dim_head=64, dropout=0.1)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=256, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)

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
        img = self.transform(org_img)

        img_f = self.conv_emb(org_img)

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.scale == 4:
            x4 = self.conv4(x3)
            x6 = self.transformer(x4)
        #elif self.scale == 5:
        #    x4 = self.conv4(x3)
        #    x5 = self.conv5(x4)
        #    x6 = self.transformer(x5)


        if self.scale == 4:
            N, C, H, W = x4.shape
        #elif self.scale == 5:
        #    N, C, H, W = x5.shape

        x6 = torch.transpose(x6, 1, 2)
        x6_global = x6[:, :, 0]
        x6_global = x6_global.unsqueeze(2).unsqueeze(3)
        x6_local = x6[:, :, 1:]
        x6_local = x6_local.reshape(N, C, H, W)

        color_mapping_global = self.color_conv_global(x6_global)
        gamma_g1 = color_mapping_global[:, :64, ]
        beta_g1 = color_mapping_global[:, 64:128, ]
        gamma_g2 = color_mapping_global[:, 128:192, ]
        beta_g2 = color_mapping_global[:, 192:256, ]
        
        gamma_g3 = color_mapping_global[:, :256:320, ]
        beta_g3 = color_mapping_global[:, 320:384, ]
        gamma_g4 = color_mapping_global[:, 384:448, ]
        beta_g4 = color_mapping_global[:, 448:512, ]
        
        



        x4 = self.conv4_1(x4)
        x3 = self.conv3_1(torch.cat((x3, F. upsample_bilinear(x4, scale_factor=2)), dim=1))
        x2 = self.conv2_1(torch.cat((x2, F. upsample_bilinear(x3, scale_factor=2)), dim=1))
        x1 = self.conv1_1(torch.cat((x1, F.upsample_bilinear(x2, scale_factor=2)), dim=1))

        _, _, ho, wo = org_img.shape
        resize = T.Resize((ho,wo))

        # x1 to original size
        x1 = resize(x1)
        gamma_l = x1[:, :64, :, :]
        beta_l = x1[:, 64:, :, :]

        # Get global transform..
        if self.glo_mode == 0:
            y = gamma_g1 * img_f + beta_g1
            y = gamma_g2 * y + beta_g2
            y = gamma_g3 * y + beta_g3
            y = gamma_g4 * y + beta_g4

        elif self.glo_mode == 1:
            # global + local
            y = (gamma_g + gamma_l) * img_f + (beta_g + beta_l)
        elif self.glo_mode == 2:
            # global , local separately
            y = gamma_g * img_f + beta_g
            y = gamma_l * y + beta_l
        elif self.glo_mode == 3:
            # local only
            y = gamma_l * img_f + beta_l

        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)


        # self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
        #                          extra_conv=True)

        N, C, H, W = color_mapping_global.shape

        y_tot = self.conv_out(y)

        # # color_mapping_global = color_mapping_global.reshape(N, C)
        # # if self.global_module == 1:
        # #     y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        # # else:
        # #     y = org_img
        # if self.residual == 1:
        #     y1 = self.res1(y)
        #     y2 = self.res2(y1)
        #     y3 = self.res3(y2)
        #     y3 = self.res4(y3)
        #     y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)
        #     y_tot = y + y1
        # else:
        #     y_tot = y
        # # Get local transform
        # # y = rx + b

        return y_tot
    
    

class DCPNet6(nn.Module):
    def __init__(self, config):
        super(DCPNet6, self).__init__()

        self.scale = config.scale
        self.act = config.act

        self.conv_emb = convBlock(input_feature=3, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)
        self.conv_out = convBlock(input_feature=64, output_feature=3, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)

        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)


        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        self.transform = T.Resize((256, 256))
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode

        #self.feature_num = self.control_point * 3 * 3 + 15
        self.feature_num = 128
        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128, output_feature=256, ksize=1, stride=1,
                                                pad=0, extra_conv=True, act=self.act)


        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m


        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma, config.trainable_offset, config.offset_param, config.offset_param2, config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                    mlp_dim=512,
                                    pool='cls', dim_head=64, dropout=0.1)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=256, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)

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
        img = self.transform(org_img)

        img_f = self.conv_emb(org_img)

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.scale == 4:
            x4 = self.conv4(x3)
            x6 = self.transformer(x4)
        #elif self.scale == 5:
        #    x4 = self.conv4(x3)
        #    x5 = self.conv5(x4)
        #    x6 = self.transformer(x5)


        if self.scale == 4:
            N, C, H, W = x4.shape
        #elif self.scale == 5:
        #    N, C, H, W = x5.shape

        x6 = torch.transpose(x6, 1, 2)
        x6_global = x6[:, :, 0]
        x6_global = x6_global.unsqueeze(2).unsqueeze(3)
        x6_local = x6[:, :, 1:]
        x6_local = x6_local.reshape(N, C, H, W)

        color_mapping_global = self.color_conv_global(x6_global)
        gamma_g1 = color_mapping_global[:, :64, ]
        beta_g1 = color_mapping_global[:, 64:128, ]
        gamma_g2 = color_mapping_global[:, 128:192, ]
        beta_g2 = color_mapping_global[:, 192:, ]



        x4 = self.conv4_1(x4)
        x3 = self.conv3_1(torch.cat((x3, F. upsample_bilinear(x4, scale_factor=2)), dim=1))
        x2 = self.conv2_1(torch.cat((x2, F. upsample_bilinear(x3, scale_factor=2)), dim=1))
        x1 = self.conv1_1(torch.cat((x1, F.upsample_bilinear(x2, scale_factor=2)), dim=1))

        _, _, ho, wo = org_img.shape
        resize = T.Resize((ho,wo))

        # x1 to original size
        x1 = resize(x1)
        gamma_l = x1[:, :64, :, :]
        beta_l = x1[:, 64:, :, :]

        # Get global transform..
        if self.glo_mode == 0:
            y = gamma_g1 * img_f ** 3 + beta_g1 * img_f ** 2 + gamma_g2 * img_f + beta_g2

        elif self.glo_mode == 1:
            # global + local
            y = (gamma_g + gamma_l) * img_f + (beta_g + beta_l)
        elif self.glo_mode == 2:
            # global , local separately
            y = gamma_g * img_f + beta_g
            y = gamma_l * y + beta_l
        elif self.glo_mode == 3:
            # local only
            y = gamma_l * img_f + beta_l

        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)


        # self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
        #                          extra_conv=True)

        N, C, H, W = color_mapping_global.shape

        y_tot = self.conv_out(y)
        
        if self.residual == 1:
            y1 = self.res1(y_tot)
            y2 = self.res2(y1)
            y3 = self.res3(y2)
            y3 = self.res4(y3)
            y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
            y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
            y1 = self.res7(y1)
            y_tot = y_tot + y1

        # # color_mapping_global = color_mapping_global.reshape(N, C)
        # # if self.global_module == 1:
        # #     y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        # # else:
        # #     y = org_img
        # if self.residual == 1:
        #     y1 = self.res1(y)
        #     y2 = self.res2(y1)
        #     y3 = self.res3(y2)
        #     y3 = self.res4(y3)
        #     y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)
        #     y_tot = y + y1
        # else:
        #     y_tot = y
        # # Get local transform
        # # y = rx + b

        return y_tot
    
    
class DCPNet7(nn.Module):
    def __init__(self, config):
        super(DCPNet7, self).__init__()

        self.scale = config.scale
        self.act = config.act

        self.conv_emb = convBlock(input_feature=3, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)
        self.conv_out = convBlock(input_feature=64, output_feature=3, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)

        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)


        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        self.transform = T.Resize((256, 256))
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode

        #self.feature_num = self.control_point * 3 * 3 + 15
        self.feature_num = 128
        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128, output_feature=256, ksize=1, stride=1,
                                                pad=0, extra_conv=True, act=self.act)


        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m


        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma, config.trainable_offset, config.offset_param, config.offset_param2, config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                    mlp_dim=512,
                                    pool='cls', dim_head=64, dropout=0.1)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=256, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        self.prelu = nn.PReLU()
        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)

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
        img = self.transform(org_img)

        img_f = self.conv_emb(org_img)

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.scale == 4:
            x4 = self.conv4(x3)
            x6 = self.transformer(x4)
        #elif self.scale == 5:
        #    x4 = self.conv4(x3)
        #    x5 = self.conv5(x4)
        #    x6 = self.transformer(x5)


        if self.scale == 4:
            N, C, H, W = x4.shape
        #elif self.scale == 5:
        #    N, C, H, W = x5.shape

        x6 = torch.transpose(x6, 1, 2)
        x6_global = x6[:, :, 0]
        x6_global = x6_global.unsqueeze(2).unsqueeze(3)
        x6_local = x6[:, :, 1:]
        x6_local = x6_local.reshape(N, C, H, W)

        color_mapping_global = self.color_conv_global(x6_global)
        gamma_g1 = color_mapping_global[:, :64, ]
        beta_g1 = color_mapping_global[:, 64:128, ]
        gamma_g2 = color_mapping_global[:, 128:192, ]
        beta_g2 = color_mapping_global[:, 192:, ]



        x4 = self.conv4_1(x4)
        x3 = self.conv3_1(torch.cat((x3, F. upsample_bilinear(x4, scale_factor=2)), dim=1))
        x2 = self.conv2_1(torch.cat((x2, F. upsample_bilinear(x3, scale_factor=2)), dim=1))
        x1 = self.conv1_1(torch.cat((x1, F.upsample_bilinear(x2, scale_factor=2)), dim=1))

        _, _, ho, wo = org_img.shape
        resize = T.Resize((ho,wo))

        # x1 to original size
        x1 = resize(x1)
        gamma_l = x1[:, :64, :, :]
        beta_l = x1[:, 64:, :, :]

        # Get global transform..
        if self.glo_mode == 0:
            y = gamma_g1 * img_f + beta_g1
            y = self.prelu(y)
            y = gamma_g2 * y + beta_g2

        elif self.glo_mode == 1:
            # global + local
            y = (gamma_g + gamma_l) * img_f + (beta_g + beta_l)
        elif self.glo_mode == 2:
            # global , local separately
            y = gamma_g * img_f + beta_g
            y = gamma_l * y + beta_l
        elif self.glo_mode == 3:
            # local only
            y = gamma_l * img_f + beta_l

        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)


        # self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
        #                          extra_conv=True)

        N, C, H, W = color_mapping_global.shape

        y_tot = self.conv_out(y)
        if self.residual == 1:
            y1 = self.res1(y_tot)
            y2 = self.res2(y1)
            y3 = self.res3(y2)
            y3 = self.res4(y3)
            y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
            y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
            y1 = self.res7(y1)
            y_tot = y_tot + y1

        # # color_mapping_global = color_mapping_global.reshape(N, C)
        # # if self.global_module == 1:
        # #     y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        # # else:
        # #     y = org_img
        # if self.residual == 1:
        #     y1 = self.res1(y)
        #     y2 = self.res2(y1)
        #     y3 = self.res3(y2)
        #     y3 = self.res4(y3)
        #     y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)
        #     y_tot = y + y1
        # else:
        #     y_tot = y
        # # Get local transform
        # # y = rx + b

        return y_tot
    
    
class DCPNet8(nn.Module):
    def __init__(self, config):
        super(DCPNet8, self).__init__()

        self.scale = config.scale
        self.act = config.act

        self.conv_emb = convBlock(input_feature=3, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)
        self.conv_out = convBlock(input_feature=64, output_feature=3, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)

        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)


        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        self.transform = T.Resize((256, 256))
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode

        #self.feature_num = self.control_point * 3 * 3 + 15
        self.feature_num = 128
        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128, output_feature=256, ksize=1, stride=1,
                                                pad=0, extra_conv=True, act=self.act)


        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m


        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma, config.trainable_offset, config.offset_param, config.offset_param2, config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                    mlp_dim=512,
                                    pool='cls', dim_head=64, dropout=0.1)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=256, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        self.prelu = nn.PReLU(num_parameters=64)
        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)

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
        img = self.transform(org_img)

        img_f = self.conv_emb(org_img)

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.scale == 4:
            x4 = self.conv4(x3)
            x6 = self.transformer(x4)
        #elif self.scale == 5:
        #    x4 = self.conv4(x3)
        #    x5 = self.conv5(x4)
        #    x6 = self.transformer(x5)


        if self.scale == 4:
            N, C, H, W = x4.shape
        #elif self.scale == 5:
        #    N, C, H, W = x5.shape

        x6 = torch.transpose(x6, 1, 2)
        x6_global = x6[:, :, 0]
        x6_global = x6_global.unsqueeze(2).unsqueeze(3)
        x6_local = x6[:, :, 1:]
        x6_local = x6_local.reshape(N, C, H, W)

        color_mapping_global = self.color_conv_global(x6_global)
        gamma_g1 = color_mapping_global[:, :64, ]
        beta_g1 = color_mapping_global[:, 64:128, ]
        gamma_g2 = color_mapping_global[:, 128:192, ]
        beta_g2 = color_mapping_global[:, 192:, ]



        x4 = self.conv4_1(x4)
        x3 = self.conv3_1(torch.cat((x3, F. upsample_bilinear(x4, scale_factor=2)), dim=1))
        x2 = self.conv2_1(torch.cat((x2, F. upsample_bilinear(x3, scale_factor=2)), dim=1))
        x1 = self.conv1_1(torch.cat((x1, F.upsample_bilinear(x2, scale_factor=2)), dim=1))

        _, _, ho, wo = org_img.shape
        resize = T.Resize((ho,wo))

        # x1 to original size
        x1 = resize(x1)
        gamma_l = x1[:, :64, :, :]
        beta_l = x1[:, 64:, :, :]

        # Get global transform..
        if self.glo_mode == 0:
            y = gamma_g1 * img_f + beta_g1
            y = self.prelu(y)
            y = gamma_g2 * y + beta_g2

        elif self.glo_mode == 1:
            # global + local
            y = (gamma_g + gamma_l) * img_f + (beta_g + beta_l)
        elif self.glo_mode == 2:
            # global , local separately
            y = gamma_g * img_f + beta_g
            y = gamma_l * y + beta_l
        elif self.glo_mode == 3:
            # local only
            y = gamma_l * img_f + beta_l

        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)


        # self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
        #                          extra_conv=True)

        N, C, H, W = color_mapping_global.shape

        y_tot = self.conv_out(y)
        
        if self.residual == 1:
            y1 = self.res1(y_tot)
            y2 = self.res2(y1)
            y3 = self.res3(y2)
            y3 = self.res4(y3)
            y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
            y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
            y1 = self.res7(y1)
            y_tot = y_tot + y1

        # # color_mapping_global = color_mapping_global.reshape(N, C)
        # # if self.global_module == 1:
        # #     y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        # # else:
        # #     y = org_img
        # if self.residual == 1:
        #     y1 = self.res1(y)
        #     y2 = self.res2(y1)
        #     y3 = self.res3(y2)
        #     y3 = self.res4(y3)
        #     y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)
        #     y_tot = y + y1
        # else:
        #     y_tot = y
        # # Get local transform
        # # y = rx + b

        return y_tot
    
def prelu_from_network(x, params):
    mask1 = x > 0
    mask2 = x <= 0
    
    x =  mask1 * x + mask2 * params * x
    
    return x
    
def prelu_from_network2(x, params):
    mask1 = x > 0
    mask2 = x <= 0
    #params = params.reshape(-1,1,1,1)
    x =  mask1 * x + mask2 * params * x
    
    return x
    
class DCPNet9(nn.Module):
    def __init__(self, config):
        super(DCPNet9, self).__init__()

        self.scale = config.scale
        self.act = config.act

        self.conv_emb = convBlock(input_feature=3, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)
        self.conv_out = convBlock(input_feature=64, output_feature=3, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)

        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)


        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        self.transform = T.Resize((256, 256))
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode

        #self.feature_num = self.control_point * 3 * 3 + 15
        self.feature_num = 128
        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128, output_feature=320, ksize=1, stride=1,
                                                pad=0, extra_conv=True, act=self.act)


        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m


        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma, config.trainable_offset, config.offset_param, config.offset_param2, config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                    mlp_dim=512,
                                    pool='cls', dim_head=64, dropout=0.1)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=256, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        self.prelu = nn.PReLU(num_parameters=64)
        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)

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
        img = self.transform(org_img)

        img_f = self.conv_emb(org_img)

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.scale == 4:
            x4 = self.conv4(x3)
            x6 = self.transformer(x4)
        #elif self.scale == 5:
        #    x4 = self.conv4(x3)
        #    x5 = self.conv5(x4)
        #    x6 = self.transformer(x5)


        if self.scale == 4:
            N, C, H, W = x4.shape
        #elif self.scale == 5:
        #    N, C, H, W = x5.shape

        x6 = torch.transpose(x6, 1, 2)
        x6_global = x6[:, :, 0]
        x6_global = x6_global.unsqueeze(2).unsqueeze(3)
        x6_local = x6[:, :, 1:]
        x6_local = x6_local.reshape(N, C, H, W)

        color_mapping_global = self.color_conv_global(x6_global)
        gamma_g1 = color_mapping_global[:, :64, ]
        beta_g1 = color_mapping_global[:, 64:128, ]
        gamma_g2 = color_mapping_global[:, 128:192, ]
        beta_g2 = color_mapping_global[:, 192:256, ]
        prelu_param = color_mapping_global[:, 256:, ]


        x4 = self.conv4_1(x4)
        x3 = self.conv3_1(torch.cat((x3, F. upsample_bilinear(x4, scale_factor=2)), dim=1))
        x2 = self.conv2_1(torch.cat((x2, F. upsample_bilinear(x3, scale_factor=2)), dim=1))
        x1 = self.conv1_1(torch.cat((x1, F.upsample_bilinear(x2, scale_factor=2)), dim=1))

        _, _, ho, wo = org_img.shape
        resize = T.Resize((ho,wo))

        # x1 to original size
        x1 = resize(x1)
        gamma_l = x1[:, :64, :, :]
        beta_l = x1[:, 64:, :, :]

        # Get global transform..
        if self.glo_mode == 0:
            y = gamma_g1 * img_f + beta_g1
            y = prelu_from_network(y, prelu_param)
            y = gamma_g2 * y + beta_g2

        elif self.glo_mode == 1:
            # global + local
            y = (gamma_g + gamma_l) * img_f + (beta_g + beta_l)
        elif self.glo_mode == 2:
            # global , local separately
            y = gamma_g * img_f + beta_g
            y = gamma_l * y + beta_l
        elif self.glo_mode == 3:
            # local only
            y = gamma_l * img_f + beta_l

        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)


        # self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
        #                          extra_conv=True)

        N, C, H, W = color_mapping_global.shape

        y_tot = self.conv_out(y)

        if self.residual == 1:
            y1 = self.res1(y_tot)
            y2 = self.res2(y1)
            y3 = self.res3(y2)
            y3 = self.res4(y3)
            y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
            y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
            y1 = self.res7(y1)
            y_tot = y_tot + y1

        
        # # color_mapping_global = color_mapping_global.reshape(N, C)
        # # if self.global_module == 1:
        # #     y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        # # else:
        # #     y = org_img
        # if self.residual == 1:
        #     y1 = self.res1(y)
        #     y2 = self.res2(y1)
        #     y3 = self.res3(y2)
        #     y3 = self.res4(y3)
        #     y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)
        #     y_tot = y + y1
        # else:
        #     y_tot = y
        # # Get local transform
        # # y = rx + b

        return y_tot
    
    
    
class DCPNet10(nn.Module):
    def __init__(self, config):
        super(DCPNet10, self).__init__()

        self.scale = config.scale
        self.act = config.act
        
        self.feature_num = config.feature_num
        self.iter_num = config.iter_num

        self.conv_emb = convBlock(input_feature=3, output_feature=self.feature_num, ksize=3, stride=1, pad=1, act=self.act)
        self.conv_out = convBlock(input_feature=self.feature_num, output_feature=3, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)

        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)


        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        self.transform = T.Resize((256, 256))
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode

        #self.feature_num = self.control_point * 3 * 3 + 15
        #self.feature_num = 128
        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128, output_feature=(self.iter_num*3-1)*self.feature_num, ksize=1, stride=1,
                                                pad=0, extra_conv=True, act=self.act)


        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m


        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma, config.trainable_offset, config.offset_param, config.offset_param2, config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                    mlp_dim=512,
                                    pool='cls', dim_head=64, dropout=0.1)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=256, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        self.prelu = nn.PReLU(num_parameters=64)
        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)

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
        img = self.transform(org_img)

        img_f = self.conv_emb(org_img)

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.scale == 4:
            x4 = self.conv4(x3)
            x6 = self.transformer(x4)
        #elif self.scale == 5:
        #    x4 = self.conv4(x3)
        #    x5 = self.conv5(x4)
        #    x6 = self.transformer(x5)


        if self.scale == 4:
            N, C, H, W = x4.shape
        #elif self.scale == 5:
        #    N, C, H, W = x5.shape

        x6 = torch.transpose(x6, 1, 2)
        x6_global = x6[:, :, 0]
        x6_global = x6_global.unsqueeze(2).unsqueeze(3)
        x6_local = x6[:, :, 1:]
        x6_local = x6_local.reshape(N, C, H, W)

        color_mapping_global = self.color_conv_global(x6_global)
        gamma_g = []
        beta_g = []
        prelu_param = []
        cur_idx = 0
        for i in range(self.iter_num):
            gamma_g.append(color_mapping_global[:, cur_idx*self.feature_num:(cur_idx+1)*self.feature_num, ])
            cur_idx += 1
            beta_g.append(color_mapping_global[:, cur_idx*self.feature_num:(cur_idx+1)*self.feature_num, ])
            cur_idx += 1
            if i != 0:
                prelu_param.append(color_mapping_global[:, cur_idx*self.feature_num:(cur_idx+1)*self.feature_num, ])
                cur_idx += 1
                
        # gamma_g1 = color_mapping_global[:, :64, ]
        # beta_g1 = color_mapping_global[:, 64:128, ]
        # gamma_g2 = color_mapping_global[:, 128:192, ]
        # beta_g2 = color_mapping_global[:, 192:256, ]
        # prelu_param = color_mapping_global[:, 256:, ]


        x4 = self.conv4_1(x4)
        x3 = self.conv3_1(torch.cat((x3, F. upsample_bilinear(x4, scale_factor=2)), dim=1))
        x2 = self.conv2_1(torch.cat((x2, F. upsample_bilinear(x3, scale_factor=2)), dim=1))
        x1 = self.conv1_1(torch.cat((x1, F.upsample_bilinear(x2, scale_factor=2)), dim=1))

        _, _, ho, wo = org_img.shape
        resize = T.Resize((ho,wo))

        # x1 to original size
        x1 = resize(x1)
        gamma_l = x1[:, :64, :, :]
        beta_l = x1[:, 64:, :, :]

        # Get global transform..
        y = img_f
        if self.glo_mode == 0:
            for i in range(self.iter_num):
                y = gamma_g[i] * y + beta_g[i]
                if i < self.iter_num - 1:
                    y = prelu_from_network(y, prelu_param[i])

        elif self.glo_mode == 1:
            # global + local
            y = (gamma_g + gamma_l) * img_f + (beta_g + beta_l)
        elif self.glo_mode == 2:
            # global , local separately
            y = gamma_g * img_f + beta_g
            y = gamma_l * y + beta_l
        elif self.glo_mode == 3:
            # local only
            y = gamma_l * img_f + beta_l

        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)


        # self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
        #                          extra_conv=True)

        N, C, H, W = color_mapping_global.shape

        y_tot = self.conv_out(y)

        if self.residual == 1:
            y1 = self.res1(y_tot)
            y2 = self.res2(y1)
            y3 = self.res3(y2)
            y3 = self.res4(y3)
            y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
            y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
            y1 = self.res7(y1)
            y_tot = y_tot + y1

        
        # # color_mapping_global = color_mapping_global.reshape(N, C)
        # # if self.global_module == 1:
        # #     y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        # # else:
        # #     y = org_img
        # if self.residual == 1:
        #     y1 = self.res1(y)
        #     y2 = self.res2(y1)
        #     y3 = self.res3(y2)
        #     y3 = self.res4(y3)
        #     y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)
        #     y_tot = y + y1
        # else:
        #     y_tot = y
        # # Get local transform
        # # y = rx + b

        return y_tot
    
    
class DCPNet11(nn.Module):
    def __init__(self, config):
        super(DCPNet11, self).__init__()

        self.scale = config.scale
        self.act = config.act
        
        self.feature_num = config.feature_num
        self.iter_num = config.iter_num

        self.conv_emb = convBlock(input_feature=3, output_feature=self.feature_num, ksize=3, stride=1, pad=1, act=self.act)
        self.conv_out = convBlock(input_feature=self.feature_num, output_feature=3, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)

        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)


        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        if config.dataset == 'ppr10ka' or config.dataset == 'ppr10kb' or config.dataset == 'ppr10kc' or config.dataset == 'adobe5k2':
            self.transform = T.Resize((448, 448))
            image_size = 448
        else:
            self.transform = T.Resize((256, 256))
            image_size = 256
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode

        #self.feature_num = self.control_point * 3 * 3 + 15
        #self.feature_num = 128
        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128, output_feature=(self.iter_num*2)*self.feature_num, ksize=1, stride=1,
                                                pad=0, extra_conv=True, act=self.act)

        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        self.prelu = []
        for i in range(self.iter_num-1):
            self.prelu.append(nn.PReLU().cuda())
        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m


        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma, config.trainable_offset, config.offset_param, config.offset_param2, config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            self.transformer = ViT2(image_size=image_size, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                    mlp_dim=512,
                                    pool='cls', dim_head=64, dropout=0.1)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=image_size, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        #self.prelu = nn.PReLU(num_parameters=64)
        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)

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
        img = self.transform(org_img)

        img_f = self.conv_emb(org_img)

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.scale == 4:
            x4 = self.conv4(x3)
            x6 = self.transformer(x4)
        #elif self.scale == 5:
        #    x4 = self.conv4(x3)
        #    x5 = self.conv5(x4)
        #    x6 = self.transformer(x5)


        if self.scale == 4:
            N, C, H, W = x4.shape
        #elif self.scale == 5:
        #    N, C, H, W = x5.shape

        x6 = torch.transpose(x6, 1, 2)
        x6_global = x6[:, :, 0]
        x6_global = x6_global.unsqueeze(2).unsqueeze(3)
        x6_local = x6[:, :, 1:]
        x6_local = x6_local.reshape(N, C, H, W)

        color_mapping_global = self.color_conv_global(x6_global)
        gamma_g = []
        beta_g = []
        cur_idx = 0
        for i in range(self.iter_num):
            gamma_g.append(color_mapping_global[:, cur_idx*self.feature_num:(cur_idx+1)*self.feature_num, ])
            cur_idx += 1
            beta_g.append(color_mapping_global[:, cur_idx*self.feature_num:(cur_idx+1)*self.feature_num, ])
            cur_idx += 1

                
        # gamma_g1 = color_mapping_global[:, :64, ]
        # beta_g1 = color_mapping_global[:, 64:128, ]
        # gamma_g2 = color_mapping_global[:, 128:192, ]
        # beta_g2 = color_mapping_global[:, 192:256, ]
        # prelu_param = color_mapping_global[:, 256:, ]


        x4 = self.conv4_1(x4)
        x3 = self.conv3_1(torch.cat((x3, F. upsample_bilinear(x4, scale_factor=2)), dim=1))
        x2 = self.conv2_1(torch.cat((x2, F. upsample_bilinear(x3, scale_factor=2)), dim=1))
        x1 = self.conv1_1(torch.cat((x1, F.upsample_bilinear(x2, scale_factor=2)), dim=1))

        _, _, ho, wo = org_img.shape
        resize = T.Resize((ho,wo))

        # x1 to original size
        x1 = resize(x1)
        gamma_l = x1[:, :64, :, :]
        beta_l = x1[:, 64:, :, :]

        # Get global transform..
        y = img_f
        if self.glo_mode == 0:
            for i in range(self.iter_num):
                y = gamma_g[i] * y + beta_g[i]
                if i < self.iter_num - 1:
                    y = self.prelu[i](y)

        elif self.glo_mode == 1:
            # global + local
            y = (gamma_g + gamma_l) * img_f + (beta_g + beta_l)
        elif self.glo_mode == 2:
            # global , local separately
            y = gamma_g * img_f + beta_g
            y = gamma_l * y + beta_l
        elif self.glo_mode == 3:
            # local only
            y = gamma_l * img_f + beta_l

        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)


        # self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
        #                          extra_conv=True)

        N, C, H, W = color_mapping_global.shape

        y_tot = self.conv_out(y)

        if self.residual == 1:
            y_tot = y_tot + org_img

        
        # # color_mapping_global = color_mapping_global.reshape(N, C)
        # # if self.global_module == 1:
        # #     y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        # # else:
        # #     y = org_img
        # if self.residual == 1:
        #     y1 = self.res1(y)
        #     y2 = self.res2(y1)
        #     y3 = self.res3(y2)
        #     y3 = self.res4(y3)
        #     y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)
        #     y_tot = y + y1
        # else:
        #     y_tot = y
        # # Get local transform
        # # y = rx + b

        return y_tot
    
    
class DCPNet12(nn.Module):
    def __init__(self, config):
        super(DCPNet12, self).__init__()

        self.scale = config.scale
        self.act = config.act
        
        self.feature_num = config.feature_num
        self.iter_num = config.iter_num

        self.conv_emb = convBlock(input_feature=3, output_feature=self.feature_num, ksize=3, stride=1, pad=1, act=self.act)
        self.conv_out = convBlock(input_feature=self.feature_num, output_feature=3, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)

        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)


        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        if config.dataset == 'ppr10ka' or config.dataset == 'ppr10kb' or config.dataset == 'ppr10kc' or config.dataset == 'adobe5k2':
            self.transform = T.Resize((448, 448))
            image_size = 448
        else:
            self.transform = T.Resize((256, 256))
            image_size = 256
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode

        #self.feature_num = self.control_point * 3 * 3 + 15
        #self.feature_num = 128
        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128, output_feature=(self.iter_num*2)*self.feature_num + (self.iter_num - 1), ksize=1, stride=1,
                                                pad=0, extra_conv=True, act=self.act)


        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m


        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma, config.trainable_offset, config.offset_param, config.offset_param2, config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            self.transformer = ViT2(image_size=image_size, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                    mlp_dim=512,
                                    pool='cls', dim_head=64, dropout=0.1)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=image_size, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        self.prelu = nn.PReLU(num_parameters=64)
        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)

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
        img = self.transform(org_img)

        img_f = self.conv_emb(org_img)

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.scale == 4:
            x4 = self.conv4(x3)
            x6 = self.transformer(x4)
        #elif self.scale == 5:
        #    x4 = self.conv4(x3)
        #    x5 = self.conv5(x4)
        #    x6 = self.transformer(x5)


        if self.scale == 4:
            N, C, H, W = x4.shape
        #elif self.scale == 5:
        #    N, C, H, W = x5.shape

        x6 = torch.transpose(x6, 1, 2)
        x6_global = x6[:, :, 0]
        x6_global = x6_global.unsqueeze(2).unsqueeze(3)
        x6_local = x6[:, :, 1:]
        x6_local = x6_local.reshape(N, C, H, W)

        color_mapping_global = self.color_conv_global(x6_global)
        gamma_g = []
        beta_g = []
        prelu_param = []
        cur_idx = 0
        for i in range(self.iter_num):
            gamma_g.append(color_mapping_global[:, cur_idx*self.feature_num:(cur_idx+1)*self.feature_num, ])
            cur_idx += 1
            beta_g.append(color_mapping_global[:, cur_idx*self.feature_num:(cur_idx+1)*self.feature_num, ])
            cur_idx += 1
        for i in range(self.iter_num-1):
            prelu_param.append(color_mapping_global[:, cur_idx*self.feature_num + i, ].reshape(-1,1,1,1))
                
        # gamma_g1 = color_mapping_global[:, :64, ]
        # beta_g1 = color_mapping_global[:, 64:128, ]
        # gamma_g2 = color_mapping_global[:, 128:192, ]
        # beta_g2 = color_mapping_global[:, 192:256, ]
        # prelu_param = color_mapping_global[:, 256:, ]


        x4 = self.conv4_1(x4)
        x3 = self.conv3_1(torch.cat((x3, F. upsample_bilinear(x4, scale_factor=2)), dim=1))
        x2 = self.conv2_1(torch.cat((x2, F. upsample_bilinear(x3, scale_factor=2)), dim=1))
        x1 = self.conv1_1(torch.cat((x1, F.upsample_bilinear(x2, scale_factor=2)), dim=1))

        _, _, ho, wo = org_img.shape
        resize = T.Resize((ho,wo))

        # x1 to original size
        x1 = resize(x1)
        gamma_l = x1[:, :64, :, :]
        beta_l = x1[:, 64:, :, :]

        # Get global transform..
        y = img_f
        if self.glo_mode == 0:
            for i in range(self.iter_num):
                y = gamma_g[i] * y + beta_g[i]
                if i < self.iter_num - 1:
                    y = prelu_from_network2(y, prelu_param[i])

        elif self.glo_mode == 1:
            # global + local
            y = (gamma_g + gamma_l) * img_f + (beta_g + beta_l)
        elif self.glo_mode == 2:
            # global , local separately
            y = gamma_g * img_f + beta_g
            y = gamma_l * y + beta_l
        elif self.glo_mode == 3:
            # local only
            y = gamma_l * img_f + beta_l

        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)


        # self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
        #                          extra_conv=True)

        N, C, H, W = color_mapping_global.shape

        y_tot = self.conv_out(y)

        if self.residual == 1:
            y1 = self.res1(y_tot)
            y2 = self.res2(y1)
            y3 = self.res3(y2)
            y3 = self.res4(y3)
            y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
            y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
            y1 = self.res7(y1)
            y_tot = y_tot + y1

        
        # # color_mapping_global = color_mapping_global.reshape(N, C)
        # # if self.global_module == 1:
        # #     y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        # # else:
        # #     y = org_img
        # if self.residual == 1:
        #     y1 = self.res1(y)
        #     y2 = self.res2(y1)
        #     y3 = self.res3(y2)
        #     y3 = self.res4(y3)
        #     y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)
        #     y_tot = y + y1
        # else:
        #     y_tot = y
        # # Get local transform
        # # y = rx + b

        return y_tot
    
    
    
class DCPNet13(nn.Module):
    def __init__(self, config):
        super(DCPNet13, self).__init__()

        self.scale = config.scale
        self.act = config.act
        
        self.feature_num = config.feature_num
        self.iter_num = config.iter_num
        self.use_transformer = config.transformer

        if config.conv_num == 1:
            self.conv_emb = convBlock(input_feature=3, output_feature=self.feature_num, ksize=3, stride=1, pad=1, act=self.act)
        else:
            lists = []
            lists.append(convBlock(input_feature=3, output_feature=self.feature_num, ksize=3, stride=1, pad=1, act=self.act))
            
            for i in range(1, config.conv_num):
                lists.append(BasicBlock(self.feature_num, self.feature_num))
                #lists.append(convBlock(input_feature=self.feature_num, output_feature=self.feature_num, ksize=3, stride=1, pad=1, act=self.act))

            self.conv_emb = nn.Sequential(*lists)
        
        
        self.conv_out = convBlock(input_feature=self.feature_num, output_feature=3, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)

        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)


        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        if config.dataset == 'ppr10ka' or config.dataset == 'ppr10kb' or config.dataset == 'ppr10kc' or config.dataset == 'adobe5k2':
            self.transform = T.Resize((448, 448))
            image_size = 448
        else:
            self.transform = T.Resize((256, 256))
            image_size = 256
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode

        #self.feature_num = self.control_point * 3 * 3 + 15
        #self.feature_num = 128
        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128, output_feature=(self.iter_num*3-1)*self.feature_num, ksize=1, stride=1,
                                                pad=0, extra_conv=True, act=self.act)


        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m


        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma, config.trainable_offset, config.offset_param, config.offset_param2, config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            if self.use_transformer == 1:
                self.transformer = ViT2(image_size=image_size, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                        mlp_dim=512,
                                        pool='cls', dim_head=64, dropout=0.1)
            else:
                lists = []
                lists.append(convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=0, act=self.act))
                lists.append(convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=0, act=self.act))
                lists.append(convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=0, act=self.act))
                lists.append(convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=0, act=self.act))
                
                lists.append(nn.AdaptiveAvgPool2d((1, 1)))
                self.transformer = nn.Sequential(*lists)
                
                output_feature = (self.iter_num*3-1)*self.feature_num
                lists2 = [nn.Conv2d(128, output_feature, kernel_size=1, stride=1, padding=0),
                                          nn.BatchNorm2d(output_feature),
                                          nn.GELU(),
                                          nn.Conv2d(output_feature, output_feature, kernel_size=1, stride=1, padding=0),
                                          nn.BatchNorm2d(output_feature),
                                          nn.GELU(),
                                          nn.Conv2d(output_feature, output_feature, kernel_size=1, stride=1, padding=0)
                                          ]
                
                self.fc1 = nn.Sequential(*lists2)
  
                
                #self.fc1 = nn.Linear(128, 128)
                #self.fc2 = nn.Linear(128, 10)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=image_size, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        self.prelu = nn.PReLU(num_parameters=64)
        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, org_img, gt, index_image, color_map_control, train):
        img = self.transform(org_img)

        img_f = []
        img_f.append(self.conv_emb(org_img))
        if train == 1:
            gt = self.transform(gt)
            gt_f = self.conv_emb(gt)
        else:
            gt_f = None

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.scale == 4:
            x4 = self.conv4(x3)
            x6 = self.transformer(x4)
        #elif self.scale == 5:
        #    x4 = self.conv4(x3)
        #    x5 = self.conv5(x4)
        #    x6 = self.transformer(x5)

        if self.use_transformer == 0:
            #x6 = x6[:,:,0,0]
            color_mapping_global = self.fc1(x6)
            #color_mapping_global = color_mapping_global[:,:,0,0]
        else:
            if self.scale == 4:
                N, C, H, W = x4.shape
            #elif self.scale == 5:
            #    N, C, H, W = x5.shape

            x6 = torch.transpose(x6, 1, 2)
            x6_global = x6[:, :, 0]
            x6_global = x6_global.unsqueeze(2).unsqueeze(3)
            x6_local = x6[:, :, 1:]
            x6_local = x6_local.reshape(N, C, H, W)

            color_mapping_global = self.color_conv_global(x6_global)
        gamma_g = []
        beta_g = []
        prelu_param = []
        cur_idx = 0
        for i in range(self.iter_num):
            gamma_g.append(color_mapping_global[:, cur_idx*self.feature_num:(cur_idx+1)*self.feature_num, ])
            cur_idx += 1
            beta_g.append(color_mapping_global[:, cur_idx*self.feature_num:(cur_idx+1)*self.feature_num, ])
            cur_idx += 1
            if i != 0:
                prelu_param.append(color_mapping_global[:, cur_idx*self.feature_num:(cur_idx+1)*self.feature_num, ])
                cur_idx += 1
                
        # gamma_g1 = color_mapping_global[:, :64, ]
        # beta_g1 = color_mapping_global[:, 64:128, ]
        # gamma_g2 = color_mapping_global[:, 128:192, ]
        # beta_g2 = color_mapping_global[:, 192:256, ]
        # prelu_param = color_mapping_global[:, 256:, ]


        x4 = self.conv4_1(x4)
        x3 = self.conv3_1(torch.cat((x3, F. upsample_bilinear(x4, scale_factor=2)), dim=1))
        x2 = self.conv2_1(torch.cat((x2, F. upsample_bilinear(x3, scale_factor=2)), dim=1))
        x1 = self.conv1_1(torch.cat((x1, F.upsample_bilinear(x2, scale_factor=2)), dim=1))

        _, _, ho, wo = org_img.shape
        resize = T.Resize((ho,wo))

        # x1 to original size
        x1 = resize(x1)
        gamma_l = x1[:, :64, :, :]
        beta_l = x1[:, 64:, :, :]

        # Get global transform..
        
        if self.glo_mode == 0:
            for i in range(self.iter_num):
                img_f.append(gamma_g[i] * img_f[i] + beta_g[i]) 
                if i < self.iter_num - 1:
                    img_f[i+1] = prelu_from_network(img_f[i+1], prelu_param[i])
        elif self.glo_mode == 1:
            # global + local
            y = (gamma_g + gamma_l) * img_f + (beta_g + beta_l)
        elif self.glo_mode == 2:
            # global , local separately
            y = gamma_g * img_f + beta_g
            y = gamma_l * y + beta_l
        elif self.glo_mode == 3:
            # local only
            y = gamma_l * img_f + beta_l

        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)


        # self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
        #                          extra_conv=True)
        
        y = img_f[self.iter_num]
        N, C, H, W = color_mapping_global.shape

        y_tot = self.conv_out(y)

        if self.residual == 1:
            y1 = self.res1(y_tot)
            y2 = self.res2(y1)
            y3 = self.res3(y2)
            y3 = self.res4(y3)
            y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
            y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
            y1 = self.res7(y1)
            y_tot = y_tot + y1

        
        # # color_mapping_global = color_mapping_global.reshape(N, C)
        # # if self.global_module == 1:
        # #     y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        # # else:
        # #     y = org_img
        # if self.residual == 1:
        #     y1 = self.res1(y)
        #     y2 = self.res2(y1)
        #     y3 = self.res3(y2)
        #     y3 = self.res4(y3)
        #     y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)
        #     y_tot = y + y1
        # else:
        #     y_tot = y
        # # Get local transform
        # # y = rx + b

        return y_tot, img_f, gt_f
    
    
    
class DCPNet14(nn.Module):
    def __init__(self, config):
        super(DCPNet14, self).__init__()

        self.scale = config.scale
        self.act = config.act
        
        self.feature_num = config.feature_num
        self.iter_num = config.iter_num

        self.conv_emb = convBlock(input_feature=3, output_feature=self.feature_num, ksize=3, stride=1, pad=1, act=self.act)
        self.conv_out = convBlock(input_feature=self.feature_num, output_feature=3, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)

        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)


        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        if config.dataset == 'ppr10ka' or config.dataset == 'ppr10kb' or config.dataset == 'ppr10kc' or config.dataset == 'adobe5k2':
            self.transform = T.Resize((448, 448))
            image_size = 448
        else:
            self.transform = T.Resize((256, 256))
            image_size = 256
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode

        #self.feature_num = self.control_point * 3 * 3 + 15
        #self.feature_num = 128
        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128, output_feature=(self.iter_num*2)*self.feature_num, ksize=1, stride=1,
                                                pad=0, extra_conv=True, act=self.act)

        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        self.prelu = []
        for i in range(self.iter_num-1):
            self.prelu.append(nn.PReLU(num_parameters=self.feature_num).cuda())
        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m


        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma, config.trainable_offset, config.offset_param, config.offset_param2, config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            self.transformer = ViT2(image_size=image_size, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                    mlp_dim=512,
                                    pool='cls', dim_head=64, dropout=0.1)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=image_size, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        #self.prelu = nn.PReLU(num_parameters=64)
        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)

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
        img = self.transform(org_img)

        img_f = self.conv_emb(org_img)

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.scale == 4:
            x4 = self.conv4(x3)
            x6 = self.transformer(x4)
        #elif self.scale == 5:
        #    x4 = self.conv4(x3)
        #    x5 = self.conv5(x4)
        #    x6 = self.transformer(x5)


        if self.scale == 4:
            N, C, H, W = x4.shape
        #elif self.scale == 5:
        #    N, C, H, W = x5.shape

        x6 = torch.transpose(x6, 1, 2)
        x6_global = x6[:, :, 0]
        x6_global = x6_global.unsqueeze(2).unsqueeze(3)
        x6_local = x6[:, :, 1:]
        x6_local = x6_local.reshape(N, C, H, W)

        color_mapping_global = self.color_conv_global(x6_global)
        gamma_g = []
        beta_g = []
        cur_idx = 0
        for i in range(self.iter_num):
            gamma_g.append(color_mapping_global[:, cur_idx*self.feature_num:(cur_idx+1)*self.feature_num, ])
            cur_idx += 1
            beta_g.append(color_mapping_global[:, cur_idx*self.feature_num:(cur_idx+1)*self.feature_num, ])
            cur_idx += 1

                
        # gamma_g1 = color_mapping_global[:, :64, ]
        # beta_g1 = color_mapping_global[:, 64:128, ]
        # gamma_g2 = color_mapping_global[:, 128:192, ]
        # beta_g2 = color_mapping_global[:, 192:256, ]
        # prelu_param = color_mapping_global[:, 256:, ]


        x4 = self.conv4_1(x4)
        x3 = self.conv3_1(torch.cat((x3, F. upsample_bilinear(x4, scale_factor=2)), dim=1))
        x2 = self.conv2_1(torch.cat((x2, F. upsample_bilinear(x3, scale_factor=2)), dim=1))
        x1 = self.conv1_1(torch.cat((x1, F.upsample_bilinear(x2, scale_factor=2)), dim=1))

        _, _, ho, wo = org_img.shape
        resize = T.Resize((ho,wo))

        # x1 to original size
        x1 = resize(x1)
        gamma_l = x1[:, :64, :, :]
        beta_l = x1[:, 64:, :, :]

        # Get global transform..
        y = img_f
        if self.glo_mode == 0:
            for i in range(self.iter_num):
                y = gamma_g[i] * y + beta_g[i]
                if i < self.iter_num - 1:
                    y = self.prelu[i](y)

        elif self.glo_mode == 1:
            # global + local
            y = (gamma_g + gamma_l) * img_f + (beta_g + beta_l)
        elif self.glo_mode == 2:
            # global , local separately
            y = gamma_g * img_f + beta_g
            y = gamma_l * y + beta_l
        elif self.glo_mode == 3:
            # local only
            y = gamma_l * img_f + beta_l

        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)


        # self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
        #                          extra_conv=True)

        N, C, H, W = color_mapping_global.shape

        y_tot = self.conv_out(y)

        if self.residual == 1:
            y1 = self.res1(y_tot)
            y2 = self.res2(y1)
            y3 = self.res3(y2)
            y3 = self.res4(y3)
            y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
            y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
            y1 = self.res7(y1)
            y_tot = y_tot + y1

        
        # # color_mapping_global = color_mapping_global.reshape(N, C)
        # # if self.global_module == 1:
        # #     y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        # # else:
        # #     y = org_img
        # if self.residual == 1:
        #     y1 = self.res1(y)
        #     y2 = self.res2(y1)
        #     y3 = self.res3(y2)
        #     y3 = self.res4(y3)
        #     y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)
        #     y_tot = y + y1
        # else:
        #     y_tot = y
        # # Get local transform
        # # y = rx + b

        return y_tot
    
    
    
class DCPNet15(nn.Module):
    def __init__(self, config):
        super(DCPNet15, self).__init__()

        self.scale = config.scale
        self.act = config.act
        
        self.feature_num = config.feature_num
        self.iter_num = config.iter_num

        self.conv_emb = convBlock(input_feature=3, output_feature=self.feature_num, ksize=3, stride=1, pad=1, act=self.act)
        self.conv_out = convBlock(input_feature=self.feature_num, output_feature=3, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)

        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)


        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        if config.dataset == 'ppr10ka' or config.dataset == 'ppr10kb' or config.dataset == 'ppr10kc' or config.dataset == 'adobe5k2':
            self.transform = T.Resize((448, 448))
            image_size = 448
        else:
            self.transform = T.Resize((256, 256))
            image_size = 256
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode

        #self.feature_num = self.control_point * 3 * 3 + 15
        #self.feature_num = 128
        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128, output_feature=(self.iter_num*2)*self.feature_num, ksize=1, stride=1,
                                                pad=0, extra_conv=True, act=self.act)

        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        self.prelu = []
        for i in range(self.iter_num * 2 - 1):
            self.prelu.append(nn.PReLU().cuda())
        self.feature_fusion = []
        for i in range(self.iter_num):
            self.feature_fusion.append(nn.Conv2d(self.feature_num, self.feature_num, kernel_size=3, stride=1, padding=1).cuda())
            
        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m


        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma, config.trainable_offset, config.offset_param, config.offset_param2, config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            self.transformer = ViT2(image_size=image_size, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                    mlp_dim=512,
                                    pool='cls', dim_head=64, dropout=0.1)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=image_size, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        #self.prelu = nn.PReLU(num_parameters=64)
        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)

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
        img = self.transform(org_img)

        img_f = self.conv_emb(org_img)

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.scale == 4:
            x4 = self.conv4(x3)
            x6 = self.transformer(x4)
        #elif self.scale == 5:
        #    x4 = self.conv4(x3)
        #    x5 = self.conv5(x4)
        #    x6 = self.transformer(x5)


        if self.scale == 4:
            N, C, H, W = x4.shape
        #elif self.scale == 5:
        #    N, C, H, W = x5.shape

        x6 = torch.transpose(x6, 1, 2)
        x6_global = x6[:, :, 0]
        x6_global = x6_global.unsqueeze(2).unsqueeze(3)
        x6_local = x6[:, :, 1:]
        x6_local = x6_local.reshape(N, C, H, W)

        color_mapping_global = self.color_conv_global(x6_global)
        gamma_g = []
        beta_g = []
        cur_idx = 0
        for i in range(self.iter_num):
            gamma_g.append(color_mapping_global[:, cur_idx*self.feature_num:(cur_idx+1)*self.feature_num, ])
            cur_idx += 1
            beta_g.append(color_mapping_global[:, cur_idx*self.feature_num:(cur_idx+1)*self.feature_num, ])
            cur_idx += 1

                
        # gamma_g1 = color_mapping_global[:, :64, ]
        # beta_g1 = color_mapping_global[:, 64:128, ]
        # gamma_g2 = color_mapping_global[:, 128:192, ]
        # beta_g2 = color_mapping_global[:, 192:256, ]
        # prelu_param = color_mapping_global[:, 256:, ]


        x4 = self.conv4_1(x4)
        x3 = self.conv3_1(torch.cat((x3, F. upsample_bilinear(x4, scale_factor=2)), dim=1))
        x2 = self.conv2_1(torch.cat((x2, F. upsample_bilinear(x3, scale_factor=2)), dim=1))
        x1 = self.conv1_1(torch.cat((x1, F.upsample_bilinear(x2, scale_factor=2)), dim=1))

        _, _, ho, wo = org_img.shape
        resize = T.Resize((ho,wo))

        # x1 to original size
        x1 = resize(x1)
        gamma_l = x1[:, :64, :, :]
        beta_l = x1[:, 64:, :, :]

        # Get global transform..
        j = 0
        y = img_f
        if self.glo_mode == 0:
            for i in range(self.iter_num):
                y = gamma_g[i] * y + beta_g[i]
                y = self.prelu[j](y)
                j = j + 1
                y = self.feature_fusion[i](y)
                if i < self.iter_num - 1:
                    y = self.prelu[j](y)
                    j = j + 1

        elif self.glo_mode == 1:
            # global + local
            y = (gamma_g + gamma_l) * img_f + (beta_g + beta_l)
        elif self.glo_mode == 2:
            # global , local separately
            y = gamma_g * img_f + beta_g
            y = gamma_l * y + beta_l
        elif self.glo_mode == 3:
            # local only
            y = gamma_l * img_f + beta_l

        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)


        # self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
        #                          extra_conv=True)

        N, C, H, W = color_mapping_global.shape

        y_tot = self.conv_out(y)

        if self.residual == 1:
            y_tot = y_tot + org_img

        
        # # color_mapping_global = color_mapping_global.reshape(N, C)
        # # if self.global_module == 1:
        # #     y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        # # else:
        # #     y = org_img
        # if self.residual == 1:
        #     y1 = self.res1(y)
        #     y2 = self.res2(y1)
        #     y3 = self.res3(y2)
        #     y3 = self.res4(y3)
        #     y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)
        #     y_tot = y + y1
        # else:
        #     y_tot = y
        # # Get local transform
        # # y = rx + b

        return y_tot
    

class DCPNet16(nn.Module):
    def __init__(self, config):
        super(DCPNet16, self).__init__()

        self.scale = config.scale
        self.act = config.act
        
        self.feature_num = config.feature_num
        self.iter_num = config.iter_num

        self.conv_emb = convBlock(input_feature=3, output_feature=self.feature_num, ksize=3, stride=1, pad=1, act=self.act)
        self.conv_out = convBlock(input_feature=self.feature_num, output_feature=3, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)

        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)


        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        if config.dataset == 'ppr10ka' or config.dataset == 'ppr10kb' or config.dataset == 'ppr10kc' or config.dataset == 'adobe5k2':
            self.transform = T.Resize((448, 448))
            image_size = 448
        else:
            self.transform = T.Resize((256, 256))
            image_size = 256
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode

        #self.feature_num = self.control_point * 3 * 3 + 15
        #self.feature_num = 128
        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128, output_feature=(self.iter_num*2)*self.feature_num, ksize=1, stride=1,
                                                pad=0, extra_conv=True, act=self.act)

        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        self.prelu = []
        for i in range(self.iter_num-1):
            self.prelu.append(nn.PReLU().cuda())
        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m


        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma, config.trainable_offset, config.offset_param, config.offset_param2, config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            self.transformer = ViT2(image_size=image_size, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                    mlp_dim=512,
                                    pool='cls', dim_head=64, dropout=0.1)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=image_size, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        #self.prelu = nn.PReLU(num_parameters=64)
        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)
        
        m = 32
        lists = []
        lists.append(nn.Conv2d(3, m, kernel_size=3, stride=2, padding=1))
        lists.append(nn.LeakyReLU(0.1))
        lists.append(nn.InstanceNorm2d(m))
        
        lists.append(nn.Conv2d(m, 2*m, kernel_size=3, stride=2, padding=1))
        lists.append(nn.LeakyReLU(0.1))
        lists.append(nn.InstanceNorm2d(2*m))
        
        lists.append(nn.Conv2d(2*m, 4*m, kernel_size=3, stride=2, padding=1))
        lists.append(nn.LeakyReLU(0.1))
        lists.append(nn.InstanceNorm2d(4*m))
        
        lists.append(nn.Conv2d(4*m, 8*m, kernel_size=3, stride=2, padding=1))
        lists.append(nn.LeakyReLU(0.1))
        lists.append(nn.InstanceNorm2d(8*m))
        
        lists.append(nn.Conv2d(8*m, 8*m, kernel_size=3, stride=2, padding=1))
        lists.append(nn.LeakyReLU(0.1))
        lists.append(nn.Dropout(p=0.5))
        lists.append(nn.AdaptiveAvgPool2d((1, 1)))
        
        lists.append(nn.Conv2d(8*m, 8*m, kernel_size=1, stride=1, padding=0))
        lists.append(nn.LeakyReLU(0.1))
        lists.append(nn.Conv2d(8*m, (self.iter_num*2)*self.feature_num, kernel_size=1, stride=1, padding=0))

        self.backbone = nn.Sequential(*lists)

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
        img = self.transform(org_img)

        img_f = self.conv_emb(org_img)


        color_mapping_global = self.backbone(img)
        gamma_g = []
        beta_g = []
        cur_idx = 0
        for i in range(self.iter_num):
            gamma_g.append(color_mapping_global[:, cur_idx*self.feature_num:(cur_idx+1)*self.feature_num, ])
            cur_idx += 1
            beta_g.append(color_mapping_global[:, cur_idx*self.feature_num:(cur_idx+1)*self.feature_num, ])
            cur_idx += 1



        # Get global transform..
        y = img_f
        if self.glo_mode == 0:
            for i in range(self.iter_num):
                y = gamma_g[i] * y + beta_g[i]
                if i < self.iter_num - 1:
                    y = self.prelu[i](y)

      
        N, C, H, W = color_mapping_global.shape

        y_tot = self.conv_out(y)

        if self.residual == 1:
            y1 = self.res1(y_tot)
            y2 = self.res2(y1)
            y3 = self.res3(y2)
            y3 = self.res4(y3)
            y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
            y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
            y1 = self.res7(y1)
            y_tot = y_tot + y1

        
        # # color_mapping_global = color_mapping_global.reshape(N, C)
        # # if self.global_module == 1:
        # #     y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        # # else:
        # #     y = org_img
        # if self.residual == 1:
        #     y1 = self.res1(y)
        #     y2 = self.res2(y1)
        #     y3 = self.res3(y2)
        #     y3 = self.res4(y3)
        #     y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)
        #     y_tot = y + y1
        # else:
        #     y_tot = y
        # # Get local transform
        # # y = rx + b

        return y_tot
    
    
class DCPNet17(nn.Module):
    def __init__(self, config):
        super(DCPNet17, self).__init__()

        self.scale = config.scale
        self.act = config.act
        
        self.feature_num = config.feature_num
        self.iter_num = config.iter_num

        self.conv_emb = convBlock(input_feature=3, output_feature=self.feature_num, ksize=3, stride=1, pad=1, act=self.act)
        self.conv_out = convBlock(input_feature=self.feature_num, output_feature=3, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)

        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)


        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        if config.dataset == 'ppr10ka' or config.dataset == 'ppr10kb' or config.dataset == 'ppr10kc' or config.dataset == 'adobe5k2':
            self.transform = T.Resize((448, 448))
            image_size = 448
        else:
            self.transform = T.Resize((256, 256))
            image_size = 256
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode

        #self.feature_num = self.control_point * 3 * 3 + 15
        #self.feature_num = 128
        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            #self.color_conv_global = convBlock2(input_feature=128, output_feature=(self.iter_num*2)*self.feature_num, ksize=1, stride=1,
            #                                    pad=0, extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128, output_feature=256, ksize=1, stride=1,
                                                pad=0, extra_conv=True, act=self.act)

        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        self.prelu = []
        for i in range(self.iter_num-1):
            self.prelu.append(nn.PReLU().cuda())
        # self.featureFusion = featureFusion()

        self.fc_global = []
        for i in range(self.iter_num):
            lists = []
            lists.append(nn.GELU().cuda())
            lists.append(nn.Conv2d(256, 2*self.feature_num, kernel_size=1, stride=1, padding=0))
            self.fc_global.append(nn.Sequential(*lists).cuda()) 
            
        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m


        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma, config.trainable_offset, config.offset_param, config.offset_param2, config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            self.transformer = ViT2(image_size=image_size, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                    mlp_dim=512,
                                    pool='cls', dim_head=64, dropout=0.1)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=image_size, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        #self.prelu = nn.PReLU(num_parameters=64)
        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)

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
        img = self.transform(org_img)

        img_f = self.conv_emb(org_img)

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.scale == 4:
            x4 = self.conv4(x3)
            x6 = self.transformer(x4)
        #elif self.scale == 5:
        #    x4 = self.conv4(x3)
        #    x5 = self.conv5(x4)
        #    x6 = self.transformer(x5)


        if self.scale == 4:
            N, C, H, W = x4.shape
        #elif self.scale == 5:
        #    N, C, H, W = x5.shape

        x6 = torch.transpose(x6, 1, 2)
        x6_global = x6[:, :, 0]
        x6_global = x6_global.unsqueeze(2).unsqueeze(3)
        x6_local = x6[:, :, 1:]
        x6_local = x6_local.reshape(N, C, H, W)

        color_mapping_global = self.color_conv_global(x6_global)
        gamma_g = []
        beta_g = []
        for i in range(self.iter_num):
            z = self.fc_global[i](color_mapping_global)
            gamma_g.append(z[:, :self.feature_num, ])
            beta_g.append(z[:,self.feature_num:,])

                
        # gamma_g1 = color_mapping_global[:, :64, ]
        # beta_g1 = color_mapping_global[:, 64:128, ]
        # gamma_g2 = color_mapping_global[:, 128:192, ]
        # beta_g2 = color_mapping_global[:, 192:256, ]
        # prelu_param = color_mapping_global[:, 256:, ]


        x4 = self.conv4_1(x4)
        x3 = self.conv3_1(torch.cat((x3, F. upsample_bilinear(x4, scale_factor=2)), dim=1))
        x2 = self.conv2_1(torch.cat((x2, F. upsample_bilinear(x3, scale_factor=2)), dim=1))
        x1 = self.conv1_1(torch.cat((x1, F.upsample_bilinear(x2, scale_factor=2)), dim=1))

        _, _, ho, wo = org_img.shape
        resize = T.Resize((ho,wo))

        # x1 to original size
        x1 = resize(x1)
        gamma_l = x1[:, :64, :, :]
        beta_l = x1[:, 64:, :, :]

        # Get global transform..
        y = img_f
        if self.glo_mode == 0:
            for i in range(self.iter_num):
                y = gamma_g[i] * y + beta_g[i]
                if i < self.iter_num - 1:
                    y = self.prelu[i](y)

        elif self.glo_mode == 1:
            # global + local
            y = (gamma_g + gamma_l) * img_f + (beta_g + beta_l)
        elif self.glo_mode == 2:
            # global , local separately
            y = gamma_g * img_f + beta_g
            y = gamma_l * y + beta_l
        elif self.glo_mode == 3:
            # local only
            y = gamma_l * img_f + beta_l

        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)


        # self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
        #                          extra_conv=True)

        N, C, H, W = color_mapping_global.shape

        y_tot = self.conv_out(y)

        if self.residual == 1:
            y1 = self.res1(y_tot)
            y2 = self.res2(y1)
            y3 = self.res3(y2)
            y3 = self.res4(y3)
            y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
            y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
            y1 = self.res7(y1)
            y_tot = y_tot + y1

        
        # # color_mapping_global = color_mapping_global.reshape(N, C)
        # # if self.global_module == 1:
        # #     y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        # # else:
        # #     y = org_img
        # if self.residual == 1:
        #     y1 = self.res1(y)
        #     y2 = self.res2(y1)
        #     y3 = self.res3(y2)
        #     y3 = self.res4(y3)
        #     y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)
        #     y_tot = y + y1
        # else:
        #     y_tot = y
        # # Get local transform
        # # y = rx + b

        return y_tot
    
    
    
class DCPNet18(nn.Module):
    def __init__(self, config):
        super(DCPNet18, self).__init__()

        self.scale = config.scale
        self.act = config.act
        
        self.feature_num = config.feature_num
        self.iter_num = config.iter_num

        self.conv_emb = convBlock(input_feature=3, output_feature=self.feature_num, ksize=3, stride=1, pad=1, act=self.act)
        self.conv_out = convBlock(input_feature=self.feature_num, output_feature=3, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)

        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act, extra_conv=True)

        self.res_mode = config.res_mode
        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        if config.dataset == 'ppr10ka' or config.dataset == 'ppr10kb' or config.dataset == 'ppr10kc' or config.dataset == 'adobe5k2':
            self.transform = T.Resize((448, 448))
            image_size = 448
        else:
            self.transform = T.Resize((256, 256))
            image_size = 256
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode

        #self.feature_num = self.control_point * 3 * 3 + 15
        #self.feature_num = 128
        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128, output_feature=(self.iter_num*2)*self.feature_num, ksize=1, stride=1,
                                                pad=0, extra_conv=True, act=self.act)

        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        self.conv_cat = convBlock2(input_feature=(self.iter_num + 1)*self.feature_num, output_feature=self.feature_num, ksize=1, stride=1,
                                                pad=0, extra_conv=False, act=self.act)
        
        self.prelu = []
        for i in range(self.iter_num-1):
            self.prelu.append(nn.PReLU().cuda())
        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m


        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma, config.trainable_offset, config.offset_param, config.offset_param2, config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            self.transformer = ViT2(image_size=image_size, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                    mlp_dim=512,
                                    pool='cls', dim_head=64, dropout=0.1)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=image_size, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        #self.prelu = nn.PReLU(num_parameters=64)
        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)

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
        img = self.transform(org_img)

        img_f = self.conv_emb(org_img)

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.scale == 4:
            x4 = self.conv4(x3)
            x6 = self.transformer(x4)
        #elif self.scale == 5:
        #    x4 = self.conv4(x3)
        #    x5 = self.conv5(x4)
        #    x6 = self.transformer(x5)


        if self.scale == 4:
            N, C, H, W = x4.shape
        #elif self.scale == 5:
        #    N, C, H, W = x5.shape

        x6 = torch.transpose(x6, 1, 2)
        x6_global = x6[:, :, 0]
        x6_global = x6_global.unsqueeze(2).unsqueeze(3)
        x6_local = x6[:, :, 1:]
        x6_local = x6_local.reshape(N, C, H, W)

        color_mapping_global = self.color_conv_global(x6_global)
        gamma_g = []
        beta_g = []
        cur_idx = 0
        for i in range(self.iter_num):
            gamma_g.append(color_mapping_global[:, cur_idx*self.feature_num:(cur_idx+1)*self.feature_num, ])
            cur_idx += 1
            beta_g.append(color_mapping_global[:, cur_idx*self.feature_num:(cur_idx+1)*self.feature_num, ])
            cur_idx += 1

                
        # gamma_g1 = color_mapping_global[:, :64, ]
        # beta_g1 = color_mapping_global[:, 64:128, ]
        # gamma_g2 = color_mapping_global[:, 128:192, ]
        # beta_g2 = color_mapping_global[:, 192:256, ]
        # prelu_param = color_mapping_global[:, 256:, ]


        x4 = self.conv4_1(x4)
        x3 = self.conv3_1(torch.cat((x3, F. upsample_bilinear(x4, scale_factor=2)), dim=1))
        x2 = self.conv2_1(torch.cat((x2, F. upsample_bilinear(x3, scale_factor=2)), dim=1))
        x1 = self.conv1_1(torch.cat((x1, F.upsample_bilinear(x2, scale_factor=2)), dim=1))

        _, _, ho, wo = org_img.shape
        resize = T.Resize((ho,wo))

        # x1 to original size
        x1 = resize(x1)
        gamma_l = x1[:, :64, :, :]
        beta_l = x1[:, 64:, :, :]

        # Get global transform..
        y = img_f
        if self.glo_mode == 0:
            if self.res_mode == 0:
                for i in range(self.iter_num):
                    y = gamma_g[i] * y + beta_g[i]
                    if i < self.iter_num - 1:
                        y = self.prelu[i](y)
            elif self.res_mode == 1:
                y_cur = y
                for i in range(self.iter_num):
                    y_cur = gamma_g[i] * y_cur + beta_g[i]
                    if i < self.iter_num - 1:
                        y_cur = self.prelu[i](y)
                    y = y + y_cur
            elif self.res_mode == 2:
                y_cat = y
                y_cur = y
                for i in range(self.iter_num):
                    y_cur = gamma_g[i] * y_cur + beta_g[i]
                    if i < self.iter_num - 1:
                        y_cur = self.prelu[i](y)
                    y_cat = torch.cat((y_cat, y_cur), dim=1)
                y = self.conv_cat(y_cat)

        elif self.glo_mode == 1:
            # global + local
            y = (gamma_g + gamma_l) * img_f + (beta_g + beta_l)
        elif self.glo_mode == 2:
            # global , local separately
            y = gamma_g * img_f + beta_g
            y = gamma_l * y + beta_l
        elif self.glo_mode == 3:
            # local only
            y = gamma_l * img_f + beta_l

        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)


        # self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
        #                          extra_conv=True)

        N, C, H, W = color_mapping_global.shape

        y_tot = self.conv_out(y)

        if self.residual == 1:
            y_tot = y_tot + org_img

        
        # # color_mapping_global = color_mapping_global.reshape(N, C)
        # # if self.global_module == 1:
        # #     y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        # # else:
        # #     y = org_img
        # if self.residual == 1:
        #     y1 = self.res1(y)
        #     y2 = self.res2(y1)
        #     y3 = self.res3(y2)
        #     y3 = self.res4(y3)
        #     y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)
        #     y_tot = y + y1
        # else:
        #     y_tot = y
        # # Get local transform
        # # y = rx + b

        return y_tot


class DCPNet19(nn.Module):
    def __init__(self, config):
        super(DCPNet19, self).__init__()

        self.scale = config.scale
        self.act = config.act

        self.feature_num = config.feature_num
        self.iter_num = config.iter_num

        self.conv_emb = convBlock(input_feature=3, output_feature=self.feature_num, ksize=3, stride=1, pad=1,
                                  act=self.act)
        self.conv_out = convBlock(input_feature=self.feature_num, output_feature=3, ksize=3, stride=1, pad=1,
                                  act=self.act, extra_conv=True)

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)

        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
                                 extra_conv=True)

        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        if config.dataset == 'ppr10ka' or config.dataset == 'ppr10kb' or config.dataset == 'ppr10kc' or config.dataset == 'adobe5k2':
            self.transform = T.Resize((448, 448))
            image_size = 448
        else:
            self.transform = T.Resize((256, 256))
            image_size = 256
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode

        # self.feature_num = self.control_point * 3 * 3 + 15
        # self.feature_num = 128
        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128,
                                                output_feature=(self.iter_num * 2) * self.feature_num, ksize=1,
                                                stride=1,
                                                pad=0, extra_conv=True, act=self.act)

        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        self.prelu = []
        for i in range(self.iter_num * 3 - 1):
            self.prelu.append(nn.PReLU().cuda())
        self.feature_fusion = []
        for i in range(self.iter_num):
            self.feature_fusion.append(nn.Conv2d(self.feature_num, 4 * self.feature_num, kernel_size=1, stride=1, padding=0).cuda())
            self.feature_fusion.append(nn.Conv2d(4 * self.feature_num, self.feature_num, kernel_size=1, stride=1, padding=0).cuda())


        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m

        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma,
                                        config.trainable_offset, config.offset_param, config.offset_param2,
                                        config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            self.transformer = ViT2(image_size=image_size, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                    mlp_dim=512,
                                    pool='cls', dim_head=64, dropout=0.1)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=image_size, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        # self.prelu = nn.PReLU(num_parameters=64)
        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)

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
        img = self.transform(org_img)

        img_f = self.conv_emb(org_img)

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.scale == 4:
            x4 = self.conv4(x3)
            x6 = self.transformer(x4)
        # elif self.scale == 5:
        #    x4 = self.conv4(x3)
        #    x5 = self.conv5(x4)
        #    x6 = self.transformer(x5)

        if self.scale == 4:
            N, C, H, W = x4.shape
        # elif self.scale == 5:
        #    N, C, H, W = x5.shape

        x6 = torch.transpose(x6, 1, 2)
        x6_global = x6[:, :, 0]
        x6_global = x6_global.unsqueeze(2).unsqueeze(3)
        x6_local = x6[:, :, 1:]
        x6_local = x6_local.reshape(N, C, H, W)

        color_mapping_global = self.color_conv_global(x6_global)
        gamma_g = []
        beta_g = []
        cur_idx = 0
        for i in range(self.iter_num):
            gamma_g.append(color_mapping_global[:, cur_idx * self.feature_num:(cur_idx + 1) * self.feature_num, ])
            cur_idx += 1
            beta_g.append(color_mapping_global[:, cur_idx * self.feature_num:(cur_idx + 1) * self.feature_num, ])
            cur_idx += 1

        # gamma_g1 = color_mapping_global[:, :64, ]
        # beta_g1 = color_mapping_global[:, 64:128, ]
        # gamma_g2 = color_mapping_global[:, 128:192, ]
        # beta_g2 = color_mapping_global[:, 192:256, ]
        # prelu_param = color_mapping_global[:, 256:, ]

        x4 = self.conv4_1(x4)
        x3 = self.conv3_1(torch.cat((x3, F.upsample_bilinear(x4, scale_factor=2)), dim=1))
        x2 = self.conv2_1(torch.cat((x2, F.upsample_bilinear(x3, scale_factor=2)), dim=1))
        x1 = self.conv1_1(torch.cat((x1, F.upsample_bilinear(x2, scale_factor=2)), dim=1))

        _, _, ho, wo = org_img.shape
        resize = T.Resize((ho, wo))

        # x1 to original size
        x1 = resize(x1)
        gamma_l = x1[:, :64, :, :]
        beta_l = x1[:, 64:, :, :]

        # Get global transform..
        j = 0
        y = img_f
        k = 0
        if self.glo_mode == 0:
            for i in range(self.iter_num):
                y = gamma_g[i] * y + beta_g[i]
                y = self.prelu[j](y)
                j = j + 1

                y = self.feature_fusion[k](y)
                y = self.prelu[j](y)
                j = j + 1
                k = k + 1
                y = self.feature_fusion[k](y)
                k = k + 1

                if i < self.iter_num - 1:
                    y = self.prelu[j](y)
                    j = j + 1

        elif self.glo_mode == 1:
            # global + local
            y = (gamma_g + gamma_l) * img_f + (beta_g + beta_l)
        elif self.glo_mode == 2:
            # global , local separately
            y = gamma_g * img_f + beta_g
            y = gamma_l * y + beta_l
        elif self.glo_mode == 3:
            # local only
            y = gamma_l * img_f + beta_l

        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)

        # self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
        #                          extra_conv=True)

        N, C, H, W = color_mapping_global.shape

        y_tot = self.conv_out(y)

        if self.residual == 1:
            y_tot = y_tot + org_img

        # # color_mapping_global = color_mapping_global.reshape(N, C)
        # # if self.global_module == 1:
        # #     y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        # # else:
        # #     y = org_img
        # if self.residual == 1:
        #     y1 = self.res1(y)
        #     y2 = self.res2(y1)
        #     y3 = self.res3(y2)
        #     y3 = self.res4(y3)
        #     y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)
        #     y_tot = y + y1
        # else:
        #     y_tot = y
        # # Get local transform
        # # y = rx + b

        return y_tot


class DCPNet20(nn.Module):
    def __init__(self, config):
        super(DCPNet20, self).__init__()

        self.scale = config.scale
        self.act = config.act

        self.feature_num = config.feature_num
        self.iter_num = config.iter_num

        self.conv_emb1 = convBlock(input_feature=3, output_feature=self.feature_num, ksize=3, stride=1, pad=1,
                                  act=self.act)
        self.conv_emb2 = convBlock(input_feature=self.feature_num, output_feature=self.feature_num, ksize=3, stride=1, pad=1,
                                   act=self.act)
        self.conv_emb3 = convBlock(input_feature=self.feature_num, output_feature=self.feature_num, ksize=3, stride=1,
                                   pad=1, act=self.act)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.conv_out1 = convBlock(input_feature=2 * self.feature_num, output_feature=3, ksize=3, stride=1, pad=1,
                                  act=self.act, extra_conv=True)

        self.conv_out2 = convBlock(input_feature=2 * self.feature_num, output_feature=self.feature_num, ksize=3, stride=1, pad=1,
                                  act=self.act, extra_conv=True)


        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)


        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        if config.dataset == 'ppr10ka' or config.dataset == 'ppr10kb' or config.dataset == 'ppr10kc' or config.dataset == 'adobe5k2':
            self.transform = T.Resize((448, 448))
            image_size = 448
        else:
            self.transform = T.Resize((256, 256))
            image_size = 256
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode


        self.color_conv_global = convBlock2(input_feature=128,
                                            output_feature=3 * (self.iter_num * 2) * self.feature_num, ksize=1,
                                            stride=1,
                                            pad=0, extra_conv=True, act=self.act)

        self.prelu = []
        for i in range(3 * (self.iter_num * 2 - 1) ):
            self.prelu.append(nn.PReLU().cuda())
        self.feature_fusion = []
        for i in range(self.iter_num):
            self.feature_fusion.append(
                nn.Conv2d(self.feature_num, self.feature_num, kernel_size=3, stride=1, padding=1).cuda())
        for i in range(self.iter_num):
            self.feature_fusion.append(
                nn.Conv2d(self.feature_num, self.feature_num, kernel_size=3, stride=1, padding=1).cuda())
        for i in range(self.iter_num):
            self.feature_fusion.append(
                nn.Conv2d(self.feature_num, self.feature_num, kernel_size=3, stride=1, padding=1).cuda())

        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m

        self.residual = config.residual

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma,
                                        config.trainable_offset, config.offset_param, config.offset_param2,
                                        config.gamma_param, config)


        self.transformer = ViT2(image_size=image_size, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                mlp_dim=512,
                                pool='cls', dim_head=64, dropout=0.1)



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
        img = self.transform(org_img)

        img_f1 = self.conv_emb1(org_img)
        img_f2 = self.conv_emb2(self.maxpool(img_f1))
        img_f3 = self.conv_emb3(self.maxpool(img_f2))

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x6 = self.transformer(x4)

        x6 = torch.transpose(x6, 1, 2)
        x6_global = x6[:, :, 0]
        x6_global = x6_global.unsqueeze(2).unsqueeze(3)

        color_mapping_global = self.color_conv_global(x6_global)
        gamma_g = []
        beta_g = []

        cur_idx = 0
        j = 0
        # scale 1
        for i in range(self.iter_num):
            gamma_g.append(color_mapping_global[:, cur_idx * self.feature_num:(cur_idx + 1) * self.feature_num, ])
            cur_idx += 1
            beta_g.append(color_mapping_global[:, cur_idx * self.feature_num:(cur_idx + 1) * self.feature_num, ])
            cur_idx += 1
        y1 = img_f1
        for i in range(self.iter_num):
            y1 = gamma_g[i] * y1 + beta_g[i]
            y1 = self.prelu[j](y1)
            j = j + 1
            y1 = self.feature_fusion[i](y1)
            if i < self.iter_num - 1:
                y1 = self.prelu[j](y1)
                j = j + 1


        # scale 2
        for i in range(self.iter_num, self.iter_num * 2):
            gamma_g.append(color_mapping_global[:, cur_idx * self.feature_num:(cur_idx + 1) * self.feature_num, ])
            cur_idx += 1
            beta_g.append(color_mapping_global[:, cur_idx * self.feature_num:(cur_idx + 1) * self.feature_num, ])
            cur_idx += 1
        y2 = img_f2
        for i in range(self.iter_num, self.iter_num * 2):
            y2 = gamma_g[i] * y2 + beta_g[i]
            y2 = self.prelu[j](y2)
            j = j + 1
            y2 = self.feature_fusion[i](y2)
            if i < self.iter_num * 2 - 1:
                y2 = self.prelu[j](y2)
                j = j + 1

        # scale 3
        for i in range(self.iter_num*2,self.iter_num*3):
            gamma_g.append(color_mapping_global[:, cur_idx * self.feature_num:(cur_idx + 1) * self.feature_num, ])
            cur_idx += 1
            beta_g.append(color_mapping_global[:, cur_idx * self.feature_num:(cur_idx + 1) * self.feature_num, ])
            cur_idx += 1
        y3 = img_f3
        for i in range(self.iter_num * 2, self.iter_num * 3):
            y3 = gamma_g[i] * y3 + beta_g[i]
            y3 = self.prelu[j](y3)
            j = j + 1
            y3 = self.feature_fusion[i](y3)
            if i < self.iter_num * 3 - 1:
                y3 = self.prelu[j](y3)
                j = j + 1

        y3 = self.upsample(y3)
        y2 = torch.cat((y2, y3), dim=1)
        y2 = self.conv_out2(y2)
        y2 = self.upsample(y2)
        y1 = torch.cat((y1, y2), dim=1)
        y_tot = self.conv_out1(y1)

        if self.residual == 1:
            y_tot = y_tot + org_img

        return y_tot


class DCPNet21(nn.Module):
    def __init__(self, config):
        super(DCPNet21, self).__init__()

        self.scale = config.scale
        self.act = config.act

        self.feature_num = config.feature_num
        self.iter_num = config.iter_num

        self.conv_emb = convBlock(input_feature=3, output_feature=self.feature_num, ksize=3, stride=1, pad=1,
                                  act=self.act)
        self.conv_out = convBlock(input_feature=self.feature_num, output_feature=3, ksize=3, stride=1, pad=1,
                                  act=self.act, extra_conv=True)

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)

        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv3_1 = convBlock(input_feature=192, output_feature=(self.iter_num * 2) * self.feature_num, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
        #                          extra_conv=True)

        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        if config.dataset == 'ppr10ka' or config.dataset == 'ppr10kb' or config.dataset == 'ppr10kc' or config.dataset == 'adobe5k2':
            self.transform = T.Resize((448, 448))
            image_size = 448
        else:
            self.transform = T.Resize((256, 256))
            image_size = 256
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode

        # self.feature_num = self.control_point * 3 * 3 + 15
        # self.feature_num = 128
        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128,
                                                output_feature=(self.iter_num * 2) * self.feature_num, ksize=1,
                                                stride=1,
                                                pad=0, extra_conv=True, act=self.act)

        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        self.prelu = []
        for i in range(self.iter_num * 4 - 1):
            self.prelu.append(nn.PReLU().cuda())
        self.feature_fusion = []
        for i in range(self.iter_num):
            self.feature_fusion.append(
                nn.Conv2d(self.feature_num, self.feature_num, kernel_size=3, stride=1, padding=1).cuda())

        self.feature_fusion2 = []
        for i in range(self.iter_num):
            self.feature_fusion2.append(
                nn.Conv2d(self.feature_num, self.feature_num, kernel_size=3, stride=1, padding=1).cuda())
        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m

        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma,
                                        config.trainable_offset, config.offset_param, config.offset_param2,
                                        config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            self.transformer = ViT2(image_size=image_size, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                    mlp_dim=512,
                                    pool='cls', dim_head=64, dropout=0.1)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=image_size, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        # self.prelu = nn.PReLU(num_parameters=64)
        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)

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
        img = self.transform(org_img)

        img_f = self.conv_emb(org_img)

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.scale == 4:
            x4 = self.conv4(x3)
            x6 = self.transformer(x4)
        # elif self.scale == 5:
        #    x4 = self.conv4(x3)
        #    x5 = self.conv5(x4)
        #    x6 = self.transformer(x5)

        if self.scale == 4:
            N, C, H, W = x4.shape
        # elif self.scale == 5:
        #    N, C, H, W = x5.shape

        x6 = torch.transpose(x6, 1, 2)
        x6_global = x6[:, :, 0]
        x6_global = x6_global.unsqueeze(2).unsqueeze(3)
        x6_local = x6[:, :, 1:]
        x6_local = x6_local.reshape(N, C, H, W)

        color_mapping_global = self.color_conv_global(x6_global)
        gamma_g = []
        beta_g = []
        cur_idx = 0
        for i in range(self.iter_num):
            gamma_g.append(color_mapping_global[:, cur_idx * self.feature_num:(cur_idx + 1) * self.feature_num, ])
            cur_idx += 1
            beta_g.append(color_mapping_global[:, cur_idx * self.feature_num:(cur_idx + 1) * self.feature_num, ])
            cur_idx += 1

        # gamma_g1 = color_mapping_global[:, :64, ]
        # beta_g1 = color_mapping_global[:, 64:128, ]
        # gamma_g2 = color_mapping_global[:, 128:192, ]
        # beta_g2 = color_mapping_global[:, 192:256, ]
        # prelu_param = color_mapping_global[:, 256:, ]

        x4 = self.conv4_1(x4)
        x3 = self.conv3_1(torch.cat((x3, F.upsample_bilinear(x4, scale_factor=2)), dim=1))

        # x2 = self.conv2_1(torch.cat((x2, F.upsample_bilinear(x3, scale_factor=2)), dim=1))
        # x1 = self.conv1_1(torch.cat((x1, F.upsample_bilinear(x2, scale_factor=2)), dim=1))

        _, _, ho, wo = org_img.shape
        resize = T.Resize((ho, wo))

        # x1 to original size
        x1 = resize(x3)
        gamma_l = []
        beta_l = []
        cur_idx = 0
        for i in range(self.iter_num):
            gamma_l.append(x1[:, cur_idx * self.feature_num:(cur_idx + 1) * self.feature_num, :, :])
            cur_idx += 1
            beta_l.append(x1[:, cur_idx * self.feature_num:(cur_idx + 1) * self.feature_num, :, :])
            cur_idx += 1

        # Get global transform..
        j = 0
        y = img_f
        if self.glo_mode == 0:
            for i in range(self.iter_num):
                y = gamma_g[i] * y + beta_g[i]
                y = self.prelu[j](y)
                j = j + 1
                y = self.feature_fusion[i](y)

                y = self.prelu[j](y)
                j = j + 1
                y = gamma_l[i] * y + beta_l[i]
                y = self.prelu[j](y)
                j = j + 1
                y = self.feature_fusion2[i](y)
                if i < self.iter_num - 1:
                    y = self.prelu[j](y)
                    j = j + 1

        elif self.glo_mode == 1:
            # global + local
            y = (gamma_g + gamma_l) * img_f + (beta_g + beta_l)
        elif self.glo_mode == 2:
            # global , local separately
            y = gamma_g * img_f + beta_g
            y = gamma_l * y + beta_l
        elif self.glo_mode == 3:
            # local only
            y = gamma_l * img_f + beta_l

        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)

        # self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
        #                          extra_conv=True)

        N, C, H, W = color_mapping_global.shape

        y_tot = self.conv_out(y)

        if self.residual == 1:
            y_tot = y_tot + org_img

        # # color_mapping_global = color_mapping_global.reshape(N, C)
        # # if self.global_module == 1:
        # #     y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        # # else:
        # #     y = org_img
        # if self.residual == 1:
        #     y1 = self.res1(y)
        #     y2 = self.res2(y1)
        #     y3 = self.res3(y2)
        #     y3 = self.res4(y3)
        #     y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)
        #     y_tot = y + y1
        # else:
        #     y_tot = y
        # # Get local transform
        # # y = rx + b

        return y_tot


class DCPNet22(nn.Module):
    def __init__(self, config):
        super(DCPNet22, self).__init__()

        self.scale = config.scale
        self.act = config.act

        self.feature_num = config.feature_num
        self.iter_num = config.iter_num

        self.conv_emb = convBlock(input_feature=3, output_feature=self.feature_num, ksize=3, stride=1, pad=1,
                                  act=self.act)
        self.conv_out = convBlock(input_feature=self.feature_num, output_feature=3, ksize=3, stride=1, pad=1,
                                  act=self.act, extra_conv=True)

        self.conv1 = convBlock(input_feature=3, output_feature=16, ksize=3, stride=2, pad=1, act=self.act)
        self.conv2 = convBlock(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)
        self.conv3 = convBlock(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)
        self.conv4 = convBlock(input_feature=64, output_feature=128, ksize=3, stride=2, pad=1, act=self.act)

        self.conv5 = convBlock(input_feature=128, output_feature=256, ksize=3, stride=2, pad=1, act=self.act)
        self.conv6 = convBlock(input_feature=256, output_feature=1024, ksize=1, stride=1, pad=0, act=self.act)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
                                 extra_conv=True)

        # no use
        self.global_conv1 = convBlock(input_feature=1024, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        # no use
        self.local_conv1 = convBlock(input_feature=64, output_feature=128, ksize=1, stride=1, pad=0, act=self.act)

        if config.dataset == 'ppr10ka' or config.dataset == 'ppr10kb' or config.dataset == 'ppr10kc' or config.dataset == 'adobe5k2':
            self.transform = T.Resize((448, 448))
            image_size = 448
        else:
            self.transform = T.Resize((256, 256))
            image_size = 256
        self.extract_f = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, extra_conv=True,
                                    act=self.act)

        self.control_point = config.control_point
        self.glo_mode = config.glo_mode

        # self.feature_num = self.control_point * 3 * 3 + 15
        # self.feature_num = 128
        if self.scale == 4:
            self.color_conv_local = convBlock2(input_feature=128, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=128,
                                                output_feature=(self.iter_num * 2) * self.feature_num, ksize=1,
                                                stride=1,
                                                pad=0, extra_conv=True, act=self.act)

        elif self.scale == 5:
            self.color_conv_local = convBlock2(input_feature=256, output_feature=24, ksize=1, stride=1, pad=0,
                                               extra_conv=True, act=self.act)
            self.color_conv_global = convBlock2(input_feature=256, output_feature=self.feature_num, ksize=1,
                                                stride=1, pad=0,
                                                extra_conv=True, act=self.act)

        self.prelu = []
        for i in range(self.iter_num * 5 - 1):
            self.prelu.append(nn.PReLU().cuda())
        self.feature_fusion = []
        for i in range(self.iter_num):
            self.feature_fusion.append(
                nn.Conv2d(self.feature_num, self.feature_num, kernel_size=1, stride=1, padding=0).cuda())
            self.feature_fusion.append(
                nn.Conv2d(self.feature_num, self.feature_num, kernel_size=1, stride=1, padding=0).cuda())
            self.feature_fusion.append(
                nn.Conv2d(self.feature_num, self.feature_num, kernel_size=1, stride=1, padding=0).cuda())
            self.feature_fusion.append(
                nn.Conv2d(self.feature_num, self.feature_num, kernel_size=1, stride=1, padding=0).cuda())

        # self.featureFusion = featureFusion()

        # no use
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)

        self.global_module = config.global_m

        self.residual = config.residual
        if self.residual == 1:
            self.res1 = convBlock2(input_feature=3, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res2 = convBlock2(input_feature=16, output_feature=32, ksize=3, stride=2, pad=1, act=self.act)  # s2
            self.res3 = convBlock2(input_feature=32, output_feature=64, ksize=3, stride=2, pad=1, act=self.act)  # s3

            self.res4 = convBlock2(input_feature=64, output_feature=64, ksize=3, stride=1, pad=1, act=self.act)  # s3
            # upsample and concat
            self.res5 = convBlock2(input_feature=96, output_feature=32, ksize=3, stride=1, pad=1, act=self.act)  # s2
            # upsample and concat
            self.res6 = convBlock2(input_feature=48, output_feature=16, ksize=3, stride=1, pad=1, act=self.act)  # s1
            self.res7 = convBlock(input_feature=16, output_feature=3, ksize=3, stride=1, pad=1, extra_conv=True,
                                  act=self.act)

        self.initialize_weights()

        self.color_map = colorTransform(self.control_point, config.use_param, config.trainable_gamma,
                                        config.trainable_offset, config.offset_param, config.offset_param2,
                                        config.gamma_param, config)
        # self.color_map = color_map_interpolation()

        if self.scale == 4:
            self.transformer = ViT2(image_size=image_size, patch_size=16, num_classes=128, dim=128, depth=2, heads=16,
                                    mlp_dim=512,
                                    pool='cls', dim_head=64, dropout=0.1)
        elif self.scale == 5:
            self.transformer = ViT2(image_size=image_size, patch_size=8, num_classes=256, dim=256, depth=2, heads=16,
                                    mlp_dim=512, pool='cls', dim_head=64, dropout=0.1)

        # self.prelu = nn.PReLU(num_parameters=64)
        # self.transformer = ViT2(image_size=256, patch_size=16, num_classes=128, dim=128, depth=2, heads=16, mlp_dim=512,
        #                         pool='cls', channels=128, dim_head=64, dropout=0.1)

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
        img = self.transform(org_img)

        img_f = self.conv_emb(org_img)

        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.scale == 4:
            x4 = self.conv4(x3)
            x6 = self.transformer(x4)
        # elif self.scale == 5:
        #    x4 = self.conv4(x3)
        #    x5 = self.conv5(x4)
        #    x6 = self.transformer(x5)

        if self.scale == 4:
            N, C, H, W = x4.shape
        # elif self.scale == 5:
        #    N, C, H, W = x5.shape

        x6 = torch.transpose(x6, 1, 2)
        x6_global = x6[:, :, 0]
        x6_global = x6_global.unsqueeze(2).unsqueeze(3)
        x6_local = x6[:, :, 1:]
        x6_local = x6_local.reshape(N, C, H, W)

        color_mapping_global = self.color_conv_global(x6_global)
        gamma_g = []
        beta_g = []
        cur_idx = 0
        for i in range(self.iter_num):
            gamma_g.append(color_mapping_global[:, cur_idx * self.feature_num:(cur_idx + 1) * self.feature_num, ])
            cur_idx += 1
            beta_g.append(color_mapping_global[:, cur_idx * self.feature_num:(cur_idx + 1) * self.feature_num, ])
            cur_idx += 1

        # gamma_g1 = color_mapping_global[:, :64, ]
        # beta_g1 = color_mapping_global[:, 64:128, ]
        # gamma_g2 = color_mapping_global[:, 128:192, ]
        # beta_g2 = color_mapping_global[:, 192:256, ]
        # prelu_param = color_mapping_global[:, 256:, ]

        x4 = self.conv4_1(x4)
        x3 = self.conv3_1(torch.cat((x3, F.upsample_bilinear(x4, scale_factor=2)), dim=1))
        x2 = self.conv2_1(torch.cat((x2, F.upsample_bilinear(x3, scale_factor=2)), dim=1))
        x1 = self.conv1_1(torch.cat((x1, F.upsample_bilinear(x2, scale_factor=2)), dim=1))

        _, _, ho, wo = org_img.shape
        resize = T.Resize((ho, wo))

        # x1 to original size
        x1 = resize(x1)
        gamma_l = x1[:, :64, :, :]
        beta_l = x1[:, 64:, :, :]

        # Get global transform..
        j = 0
        y = img_f
        k = 0
        if self.glo_mode == 0:
            for i in range(self.iter_num):
                y = gamma_g[i] * y + beta_g[i]
                y = self.prelu[j](y)
                j = j + 1

                y = self.feature_fusion[k](y)
                y = self.prelu[j](y)
                j = j + 1
                k = k + 1

                y = self.feature_fusion[k](y)
                y = self.prelu[j](y)
                j = j + 1
                k = k + 1

                y = self.feature_fusion[k](y)
                y = self.prelu[j](y)
                j = j + 1
                k = k + 1

                y = self.feature_fusion[k](y)
                k = k + 1
                if i < self.iter_num - 1:
                    y = self.prelu[j](y)
                    j = j + 1

        elif self.glo_mode == 1:
            # global + local
            y = (gamma_g + gamma_l) * img_f + (beta_g + beta_l)
        elif self.glo_mode == 2:
            # global , local separately
            y = gamma_g * img_f + beta_g
            y = gamma_l * y + beta_l
        elif self.glo_mode == 3:
            # local only
            y = gamma_l * img_f + beta_l

        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)

        # self.conv4_1 = convBlock(input_feature=128, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv3_1 = convBlock(input_feature=192, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv2_1 = convBlock(input_feature=160, output_feature=128, ksize=3, stride=1, pad=1, act=self.act)
        # self.conv1_1 = convBlock(input_feature=144, output_feature=128, ksize=3, stride=1, pad=1, act=self.act,
        #                          extra_conv=True)

        N, C, H, W = color_mapping_global.shape

        y_tot = self.conv_out(y)

        if self.residual == 1:
            y_tot = y_tot + org_img

        # # color_mapping_global = color_mapping_global.reshape(N, C)
        # # if self.global_module == 1:
        # #     y = self.color_map(org_img, color_mapping_global, index_image, color_map_control)
        # # else:
        # #     y = org_img
        # if self.residual == 1:
        #     y1 = self.res1(y)
        #     y2 = self.res2(y1)
        #     y3 = self.res3(y2)
        #     y3 = self.res4(y3)
        #     y2 = self.res5(torch.cat((y2, F.upsample_bilinear(y3, scale_factor=2)), dim=1))
        #     y1 = self.res6(torch.cat((y1, F.upsample_bilinear(y2, scale_factor=2)), dim=1))
        #     y1 = self.res7(y1)
        #     y_tot = y + y1
        # else:
        #     y_tot = y
        # # Get local transform
        # # y = rx + b

        return y_tot


class DCPNet23(nn.Module):
    def __init__(self, config):
        super(DCPNet23, self).__init__()

        self.control_point_num = config.control_point + 2
        self.feature_num = config.feature_num
        self.classifier = resnet18_224(out_dim=self.control_point_num * self.feature_num)
        self.params = nn.Parameter(torch.randn(self.feature_num, 3, 1, 1))

        self.colorTransform = colorTransform2(self.control_point_num, config.offset_param, config)

        #self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=1, stride=1, padding=0).cuda()
        self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=3, stride=1, padding=1).cuda()


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
        #out_img = torch.zeros(N,C,H,W).cuda()
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

        out_img_reshaped = img_reshaped_coeff_one * mapped_color_map_control_y + img_reshaped_coeff * mapped_color_map_control_y_plus
        # for i in range(0, self.control_point):
        #     mask = img_reshaped_index == i
        #     masked_img_reshaped_coeff = mask * img_reshaped_coeff
        #     masked_img_reshaped_coeff_one = mask * img_reshaped_coeff_one
        #     out_img_reshaped += masked_img_reshaped_coeff_one * color_map_control_y[:,:,i:i+1] + masked_img_reshaped_coeff * color_map_control_y[:,:,i+1:i+2]

        out_img_reshaped = out_img_reshaped.reshape(N, C, H, W)
        return out_img_reshaped



class DCPNet24(nn.Module):
    def __init__(self, config):
        super(DCPNet24, self).__init__()

        self.control_point_num = config.control_point + 2
        self.feature_num = config.feature_num
        
        self.hyper = config.hyper
        if self.hyper == 0:
            self.classifier = resnet18_224(out_dim=self.control_point_num * self.feature_num)
            self.params = nn.Parameter(torch.randn(self.feature_num, 3, 1, 1))
        elif self.hyper == 1:
            self.classifier = resnet18_224(out_dim=self.control_point_num * self.feature_num + 3 * self.feature_num)

        

        self.colorTransform = colorTransform3(self.control_point_num, config.offset_param, config)

        #self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=1, stride=1, padding=0).cuda()
        self.conv_out = nn.Conv2d(self.feature_num, 3, kernel_size=3, stride=1, padding=1).cuda()


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

        if self.hyper == 0:
            norm_params = self.sigmoid(self.params)
            epsilon = 1e-10
            w_sum = torch.sum(norm_params, dim=1, keepdim=True)
            norm_params = norm_params / (w_sum + epsilon)
            img_f = F.conv2d(input=org_img, weight=norm_params)
            img_f_t = self.colorTransform(img_f, self.cls_output, index_image, color_map_control)
        elif self.hyper == 1:
            offset_param = self.cls_output[:,:self.control_point_num * self.feature_num]
            transform_params = self.cls_output[:,self.control_point_num * self.feature_num:]
            transform_params = transform_params.reshape(-1, self.feature_num, 3)
            norm_params = self.sigmoid(transform_params)
            epsilon = 1e-10
            w_sum = torch.sum(norm_params, dim=1, keepdim=True)
            norm_params = norm_params / (w_sum + epsilon)
            img_f = F.conv2d(input=org_img, weight=norm_params)
            img_f_t = self.colorTransform(img_f, offset_param, index_image, color_map_control)

        
        #self.conv_emb = F.conv2d(3, self.feature_num, weight=norm_params, kernel_size=1, stride=1, padding=0, bias=False)
        #self.temp_weight =
        #conv_emb = nn.Conv2d(3, self.feature_num, weight= , kernel_size=1, stride=1, padding=0, bias=False)
        out_img = self.conv_out(img_f_t)

        #img_f = self.conv_emb(org_img)

        return out_img


class colorTransform3(nn.Module):
    def __init__(self, control_point=16, offset_param=0.04, config=0):
        super(colorTransform3, self).__init__()
        self.w = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.softmax = nn.Softmax(dim=0)
        self.control_point = control_point
        self.sigmoid = torch.nn.Sigmoid()
        self.config = config
        self.feature_num = config.feature_num

        self.epsilon = 1e-8

        self.offset_param = nn.Parameter(torch.tensor([offset_param], dtype=torch.float32))



    def forward(self, org_img, params, color_mapping_global_a, color_map_control):
        #out_img = torch.zeros(N,C,H,W).cuda()
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

        out_img_reshaped = img_reshaped_coeff_one * mapped_color_map_control_y + img_reshaped_coeff * mapped_color_map_control_y_plus
        # for i in range(0, self.control_point):
        #     mask = img_reshaped_index == i
        #     masked_img_reshaped_coeff = mask * img_reshaped_coeff
        #     masked_img_reshaped_coeff_one = mask * img_reshaped_coeff_one
        #     out_img_reshaped += masked_img_reshaped_coeff_one * color_map_control_y[:,:,i:i+1] + masked_img_reshaped_coeff * color_map_control_y[:,:,i+1:i+2]

        out_img_reshaped = out_img_reshaped.reshape(N, C, H, W)
        return out_img_reshaped

class resnet18_224(nn.Module):

    def __init__(self, out_dim=5, aug_test=False):
        super(resnet18_224, self).__init__()

        self.aug_test = aug_test
        net = models.resnet18(pretrained=True)
        # self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        # self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear')
        net.fc = nn.Linear(512, out_dim)
        self.model = net

    def forward(self, x):
        x = self.upsample(x)
        if self.aug_test:
            # x = torch.cat((x, torch.rot90(x, 1, [2, 3]), torch.rot90(x, 3, [2, 3])), 0)
            x = torch.cat((x, torch.flip(x, [3])), 0)
        f = self.model(x)

        return f
