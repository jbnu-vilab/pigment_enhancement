import os


import data_loader
import time
import sys
import torch.nn as nn
import kornia
import math
import torch
from scheduler import WarmupCosineSchedule, ConstantLRSchedule
from torchvision.utils import save_image
from vggloss import VGGPerceptualLoss, VGGContrastiveLoss
import lpips
import models_proposed

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 작업 그룹 초기화
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def gram_matrix(input):
    a, b, c, d = input.size()  # a=배치 크기(=1)
    # b=특징 맵의 수
    # (c,d)=특징 맵의 차원 (N=c*d)

    features = input.view(a * b, c * d)  # F_XL을 \hat F_XL로 크기 조정

    G = torch.mm(features, features.t())  # gram product를 계산

    # 각 특징 맵이 갖는 값의 수로 나누어
    # gram 행렬의 값을 '정규화'
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        loss = torch.nn.functional.l1_loss(G, self.target)
        #self.loss = F.mse_loss(G, self.target)
        return loss
    
import pytorch_msssim
class PSNR(nn.Module):
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        super(PSNR, self).__init__()
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        outputs = torch.unbind(torch.clamp(img1, 0, 1), dim=0)
        targets = torch.unbind(torch.clamp(img2, 0, 1), dim=0)
        psnr = [kornia.metrics.psnr(y, t, max_val=1.0) for y, t in zip(outputs, targets)]
        psnr = torch.mean(torch.stack(psnr, dim=0))
        return psnr


def calculate_delta_lab(outputs, targets):
    outputs_lab = kornia.color.rgb_to_lab(outputs)
    targets_lab = kornia.color.rgb_to_lab(targets)

    diff = (outputs_lab - targets_lab) ** 2

    delta_e = torch.mean(torch.sqrt(torch.sum(diff, dim=1, keepdim=False)))
    return delta_e

def structural_similarity(outputs, targets):
    outputs = torch.clamp(outputs, 0, 1)
    targets = torch.clamp(targets, 0, 1)

    ssim = kornia.metrics.ssim(outputs, targets, window_size=11, max_val=1.0, padding='valid')
    ssim = torch.mean(ssim)
    return ssim

class total_variation_loss(nn.Module):
    """Compute the Total Variation
    Args:
        targets: the image with shape N x C x H x W (torch.Tensor)
        outputs: the image with shape N x C x H x W (torch.Tensor)

    Return:
         a scalar with the computer loss
    """
    def __init__(self):
        super(total_variation_loss, self).__init__()
        self.name = "tv"

    @staticmethod
    def __call__(outputs, targets, gamma = 1):
        assert len(targets.shape) == 4
        assert len(outputs.shape) == 4

        target_dx = torch.abs(targets[..., 1:, :] - targets[..., :-1, :])
        target_dy = torch.abs(targets[..., :, 1:] - targets[..., :, :-1])
        weight_x = torch.exp(-gamma * torch.sum(target_dx, dim=1))
        weight_y = torch.exp(-gamma * torch.sum(target_dy, dim=1))

        output_dx = torch.abs(outputs[..., 1:, :] - outputs[..., :-1, :])
        output_dy = torch.abs(outputs[..., :, 1:] - outputs[..., :, :-1])
        loss_x = torch.mean(weight_x * torch.sum(output_dx, dim=1))
        loss_y = torch.mean(weight_y * torch.sum(output_dy, dim=1))
        loss = loss_x + loss_y
        return loss

def tempSave(org, pred, label, index):
    img = pred[index]
    img_path2 = "a.png"
    save_image(img, img_path2)
    img2 = label[index]

    img_path3 = "b.png"
    save_image(img2, img_path3)

    img3 = org[index]
    img_path3 = "c.png"
    save_image(img3, img_path3)

class solver_IE(object):
    """Solver for training and testing"""
    def __init__(self, config, path):
        self.epochs = config.epochs
        self.log = config.logs
        self.dataset = config.dataset
        self.saveimg = config.saveimg
        self.test_step = config.test_step
        self.vgg_loss = config.vgg_loss
        self.tv = config.total_variation
        self.vgg =config.vgg
        self.norm = config.norm
        self.model = config.model
        self.iter_num = config.iter_num
        self.weight_mode = config.weight_mode
        self.style_loss = config.style_loss
        self.parallel = config.parallel
        self.modeln = config.model

        if config.parallel > 0:
            self.rank = config.rank
            device = config.rank
        else:
            config.rank = 0
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        if config.loss == 'l1':
            self.l1_loss = torch.nn.L1Loss().cuda(device)
        elif config.loss == 'l2':
            self.l1_loss = torch.nn.MSELoss().cuda(device)

        self.lpips_fn = lpips.LPIPS().cuda(device)


        

        self.config = config
        self.vgg_mode = config.vgg_mode
        self.contrastive = config.contrastive
        if self.contrastive == 0:
            self.vgg_criterion = VGGPerceptualLoss(self.vgg_mode).to(device)
        elif self.contrastive == 1:
            self.vgg_criterion = VGGContrastiveLoss(config.vgg_mode).to(device)
        #
        self.total_variation_loss = total_variation_loss().to(device)

        #self.model = DCPNet(config).cuda()
        if config.model == 23:
            self.model = models_proposed.DCPNet23(config).cuda()
        elif config.model == 24:
            self.model = models_proposed.DCPNet24(config).cuda()
        elif config.model == 25:
            self.model = models_proposed.DCPNet25(config).cuda()
        elif config.model == 26:
            self.model = models_proposed.DCPNet26(config).cuda()
        elif config.model == 27:
            self.model = models_proposed.DCPNet27(config).cuda()
        elif config.model == 28:
            self.model = models_proposed.DCPNet28(config).cuda()
        elif config.model == 29:
            self.model = models_proposed.DCPNet29_cor(config).cuda()
        elif config.model == 30:
            self.model = models_proposed.DCPNet30(config).cuda()
        elif config.model == 240:
            self.model = models_proposed.DCPNet240(config).cuda()
            
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        pytorch_total_params2 = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        pytorch_total_params3 = sum(p.numel() for p in self.model.classifier.parameters())
        pytorch_total_params4 = sum(p.numel() for p in self.model.mid_conv_module.parameters())
        
        if config.parallel > 0:
            self.model = self.model.cuda(config.rank)
            self.model = DDP(module=self.model, find_unused_parameters=True, device_ids=[config.rank])
        self.PSNR = PSNR().cuda(device)
        self.PSNR.training = False
        self.lr = config.lr
        self.lrratio = config.lrratio
        self.weight_decay = config.weight_decay


        train_loader = data_loader.DataLoader(config.dataset, path, config=config, batch_size=config.batch_size, istrain=True, num_workers=config.num_workers)
        test_loader = data_loader.DataLoader(config.dataset, path, config=config, batch_size=1, istrain=False)
        if self.parallel > 0:
            self.train_sampler = train_loader.train_sampler

        batch_step_num = math.ceil(train_loader.data.__len__() / config.batch_size)

        self.new_res = config.new_res
        self.hyper = config.hyper
        if config.new_res > 0:
            if config.parallel == 0:
                backbone_params = list(map(id, self.model.classifier.model.parameters()))
                other_params = filter(lambda p: id(p) not in backbone_params, self.model.parameters())
                self.paras = [{'params': other_params, 'lr': self.lr}, {'params': self.model.classifier.model.parameters(), 'lr': self.lr * config.param1_lr_ratio}]
            else:
                backbone_params = list(map(id, self.model.module.classifier.model.parameters()))
                other_params = filter(lambda p: id(p) not in backbone_params, self.model.module.parameters())
                self.paras = [{'params': other_params, 'lr': self.lr}, {'params': self.model.module.classifier.model.parameters(), 'lr': self.lr * config.param1_lr_ratio}]
            
            self.optimizer = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)
            # backbone만 lr ratio 줘서 해보기...
            # if config.model == 29:
            #     self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            # else:
            #     self.param1_freeze_epoch = config.param1_freeze_epoch
            #     self.param2_freeze_epoch = config.param2_freeze_epoch
            #     if config.parallel == 0:
            #         param1_params = list(map(id, self.model.classifier.fc.parameters()))
            #     else:
            #         param1_params = list(map(id, self.model.module.classifier.fc.parameters()))
            #     if config.hyper > 0:
            #         if config.parallel == 0:
            #             param2_params = list(map(id, self.model.classifier.fc2.parameters()))
            #         else:
            #             param2_params = list(map(id, self.model.module.classifier.fc2.parameters()))
            #         param1_params += param2_params
            #         other_params = filter(lambda p: id(p) not in param1_params, self.model.parameters())
            #         if config.parallel == 0:
            #             self.paras = [{'params': other_params, 'lr': self.lr},
            #                         {'params': self.model.classifier.fc.parameters(), 'lr': self.lr * config.param1_lr_ratio},
            #                         {'params': self.model.classifier.fc2.parameters(), 'lr': self.lr * config.param2_lr_ratio}]
            #         else:
            #             self.paras = [{'params': other_params, 'lr': self.lr},
            #                         {'params': self.model.module.classifier.fc.parameters(), 'lr': self.lr * config.param1_lr_ratio},
            #                         {'params': self.model.module.classifier.fc2.parameters(), 'lr': self.lr * config.param2_lr_ratio}]
            #     else:
            #         other_params = filter(lambda p: id(p) not in param1_params, self.model.parameters())
            #         self.paras = [{'params': other_params, 'lr': self.lr},
            #                     {'params': self.model.module.classifier.fc.parameters(), 'lr': self.lr * config.param1_lr_ratio}]
            #     self.optimizer = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)
            if self.config.optimizer_debug == 1:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        #backbone_params = list(map(id, self.model.classifier.parameters()))
        #self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model.parameters())
        #self.paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
        #              {'params': self.model.classifier.parameters(), 'lr': self.lr}
        #              ]
        #self.optimizer = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)



        if config.scheduler == 'cos_warmup':
            self.scheduler = WarmupCosineSchedule(self.optimizer, warmup_steps=math.ceil(batch_step_num * config.warmup_step), t_total=batch_step_num * config.epochs, cycles=0.5)
        elif config.scheduler == 'constant':
            self.scheduler = ConstantLRSchedule(self.optimizer)
        elif config.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs * batch_step_num, eta_min = 5e-10)
        
        if config.resume == 1: # resume to the latest epoch
            if self.parallel > 0:
                checkpoint = torch.load('./model/{}_latest.pth'.format(self.log[:-4]))
            else:
                checkpoint = torch.load('./model/{}_latest.pth'.format(self.log[:-4]))
                #checkpoint = torch.load('./model/{}_latest.pth'.format(self.log[:-4]), map_location='cuda:0')

            self.model.load_state_dict(checkpoint["model"], strict=False)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_epoch = checkpoint["epoch"]
            self.best_psnr = checkpoint["best_psnr"]
            self.best_loss = checkpoint["best_loss"]
            self.best_ssim = checkpoint["best_ssim"]
            self.best_lpips = checkpoint["best_lpips"]
            self.best_epoch = checkpoint["best_epoch"]
            if 'best_delta_lab' in checkpoint.keys():
                self.best_delta_lab = checkpoint["best_delta_lab"]
            else:
                self.best_delta_lab = 100
            print(self.start_epoch, self.best_psnr, self.best_loss, self.best_ssim, self.best_lpips)
        elif config.resume == 2: # resume to the best epoch
            if self.parallel > 0:
                checkpoint = torch.load('./model/{}_best.pth'.format(self.log[:-4]))
            else:
                checkpoint = torch.load('./model/{}_best.pth'.format(self.log[:-4]))
                #checkpoint = torch.load('./model/{}_best.pth'.format(self.log[:-4]), map_location='cuda:0')

            self.model.load_state_dict(checkpoint["model"], strict=False)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_epoch = checkpoint["epoch"]
            self.best_psnr = checkpoint["best_psnr"]
            self.best_loss = checkpoint["best_loss"]
            self.best_ssim = checkpoint["best_ssim"]
            self.best_lpips = checkpoint["best_lpips"]
            self.best_epoch = checkpoint["best_epoch"]
            if 'best_delta_lab' in checkpoint.keys():
                self.best_delta_lab = checkpoint["best_delta_lab"]
            else:
                self.best_delta_lab = 100

        else:
            self.start_epoch = 0
            self.best_psnr = 0
            self.best_loss = 100
            self.best_ssim = 0
            self.best_lpips = 100
            self.best_delta_lab = 100

            self.best_epoch = 0

        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

        self.f = config.f
        self.control_point = config.control_point

    def train(self):
        best_psnr = self.best_psnr
        best_loss = self.best_loss
        best_ssim = self.best_ssim
        best_lpips = self.best_lpips
        best_delta_lab = self.best_delta_lab

        best_psnr2 = self.best_psnr
        best_loss2 = self.best_loss
        best_ssim2 = self.best_ssim
        best_lpips2 = self.best_lpips
        best_delta_lab2 = self.best_delta_lab

        best_psnr3 = self.best_psnr
        best_loss3 = self.best_loss
        best_ssim3 = self.best_ssim
        best_lpips3 = self.best_lpips
        best_delta_lab3 = self.best_delta_lab

        best_epoch = self.best_epoch
        device = self.device

        for t in range(self.start_epoch, self.epochs):
            # if self.config.model != 29:
            #     if self.parallel == 0:
            #         if self.new_res > 0:
            #             if t+1 == self.param1_freeze_epoch:
            #                 for name, param in self.model.classifier.fc.named_parameters():
            #                     param.requires_grad = False
            #             if self.hyper > 0:
            #                 if t + 1 == self.param2_freeze_epoch:
            #                     for name, param in self.model.classifier.fc2.named_parameters():
            #                         param.requires_grad = False
            #     else:
            #         if self.new_res > 0:
            #             if t+1 == self.param1_freeze_epoch:
            #                 for name, param in self.model.module.classifier.fc.named_parameters():
            #                     param.requires_grad = False
            #             if self.hyper > 0:
            #                 if t + 1 == self.param2_freeze_epoch:
            #                     for name, param in self.model.module.classifier.fc2.named_parameters():
            #                         param.requires_grad = False
            epoch_loss = 0
            epoch_psnr = 0
            epoch_ssim = 0
            epoch_lpips = 0
            epoch_delta_lab = 0
            i = 0
            #if t - best_epoch >= 200:
            #    break
            if self.parallel > 0:
                self.train_sampler.set_epoch(t)
            start = time.time()
            for img, label, index, img_idx in self.train_data:
                i = i+1
                N, C, H, W = img.shape
                temp = [i / (self.control_point+1) for i in range(self.control_point+2)]
                color_position = torch.tensor(temp)
                color_position = color_position.unsqueeze(0).unsqueeze(1)
                color_position = color_position.repeat(N, self.config.feature_num, 1)
                
                #color_position = torch.tensor(color_position.cuda(device))

                #img = torch.tensor(img.cuda(device))
                #index = torch.tensor(index.cuda(device))
                #label = torch.tensor(label.cuda(device))
                
                color_position = color_position.cuda(device)

                img = img.cuda(device)
                index = index.cuda(device)
                label = label.cuda(device)
                self.optimizer.zero_grad()

                
                if self.weight_mode == 0:
                    loss_weight = [0, 0, 0]
                elif self.weight_mode == 1:
                    loss_weight = [0.05, 0.1, 0.3]
                elif self.weight_mode == 2:
                    loss_weight = [0, 0.1, 0.3]
                elif self.weight_mode == 3:
                    loss_weight = [0.1, 0.3, 0.5]
                elif self.weight_mode == 4:
                    loss_weight = [0, 0, 0.3]
                elif self.weight_mode == 5:
                    loss_weight = [0, 0, 0.1]
                elif self.weight_mode == 6:
                    loss_weight = [0.02, 0.05, 0.1]

                if self.modeln == 30 or self.modeln == 31:
                    pred, params = self.model(img, index, color_position)
                else:
                    pred = self.model(img, index, color_position)
                if self.vgg_loss == 0:
                    loss = self.l1_loss(pred, label)
                else:
                    loss = self.l1_loss(pred, label) + self.vgg * self.vgg_criterion(pred, label)
                if self.modeln == 30 or self.modeln == 31:
                    if self.config.model_loss > 0:
                        if self.config.model_loss_type == 1:
                            params = torch.abs(params)
                            loss += (self.config.model_loss * torch.mean(params))
                        elif self.config.model_loss_type == 2:
                            #consistency
                            params_shift_left = params[:,:,1:]
                            params_org = params[:,:,:-1]
                            diff = torch.abs(params_shift_left - params_org)
                            loss += (self.config.model_loss * torch.mean(diff))
                        elif self.config.model_loss_type == 3:
                            #monotonicity
                            params_shift_left = params[:, :, 1:]
                            params_org = params[:, :, :-1]
                            m = nn.ReLU()
                            diff = m(params_org - params_shift_left)
                            loss += (self.config.model_loss * torch.mean(diff))
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                with torch.no_grad():
                    epoch_loss = epoch_loss + loss.detach().cpu().numpy()

                    if self.norm == 1:
                        pred = 0.5 * (pred + 1.0)
                        label = 0.5 * (label + 1.0)
                    psnr = self.PSNR(pred, label)

                    ssim = structural_similarity(pred, label)

                    delta_lab = calculate_delta_lab(pred, label)

                    pred_lpips = pred.detach() * 2.0 - 1.0
                    label_lpips = label.detach() * 2.0 - 1.0
                    lpips = torch.mean(self.lpips_fn(pred_lpips, label_lpips).squeeze())

                    epoch_psnr = epoch_psnr + psnr.detach().cpu().numpy()
                    epoch_ssim = epoch_ssim + ssim.detach().cpu().numpy()
                    epoch_lpips = epoch_lpips + lpips.detach().cpu().numpy()
                    epoch_delta_lab = epoch_delta_lab + delta_lab.detach().cpu().numpy()

                    if i % 20 == 1:
                        sys.stdout.write('\rEpoch {}: {}/{}, loss: {}'.format(t + 1, i, self.train_data.__len__(), loss))
                        self.f.write('Epoch {}: {}/{}, loss: {}\n'.format(t + 1, i, self.train_data.__len__(), loss))

            end = time.time()
            print('\n%f sec\n' % (end-start) )
            self.f.write('\n%f sec\n' % (end-start) )
            epoch_loss = epoch_loss / i
            epoch_psnr = epoch_psnr / i
            epoch_ssim = epoch_ssim / i
            epoch_lpips = epoch_lpips / i
            epoch_delta_lab = epoch_delta_lab / i
            print('train loss %f, PSNR %f SSIM %f LPIPS %f Delta_LAB %f' % (epoch_loss, epoch_psnr, epoch_ssim, epoch_lpips, epoch_delta_lab))
            self.f.write('train loss %f, PSNR %f SSIM %f LPIPS %f Delta_LAB %f\n' % (epoch_loss, epoch_psnr, epoch_ssim, epoch_lpips, epoch_delta_lab))
            if (t+1) % self.test_step == 0:
                with torch.no_grad():
                    if self.parallel == 1:
                        #print("rank: {}\n".format(self.config.rank))
                        if self.config.rank != 0:
                            continue
                        #print("rank: {}\n".format(self.config.rank))
                    test_loss, test_psnr, test_ssim, test_lpips, test_delta_lab = self.test(self.test_data)
                    print('test loss %f, PSNR %f SSIM %f LPIPS %f Delta_LAB %f' % (test_loss, test_psnr, test_ssim, test_lpips, test_delta_lab))
                    self.f.write('test loss %f, PSNR %f SSIM %f LPIPS %f Delta_LAB %f\n' % (test_loss, test_psnr, test_ssim, test_lpips, test_delta_lab))

                    if best_psnr < test_psnr:
                        best_psnr = test_psnr
                        best_loss = test_loss
                        best_ssim = test_ssim
                        best_lpips = test_lpips
                        best_delta_lab = test_delta_lab
                        best_epoch = t

                    if best_ssim2 < test_ssim:
                        best_psnr2 = test_psnr
                        best_loss2 = test_loss
                        best_ssim2 = test_ssim
                        best_lpips2 = test_lpips
                        best_delta_lab2 = test_delta_lab

                    if best_delta_lab3 > test_delta_lab:
                        best_psnr3 = test_psnr
                        best_loss3 = test_loss
                        best_ssim3 = test_ssim
                        best_lpips3 = test_lpips
                        best_delta_lab3 = test_delta_lab



                    save_dict = {
                        "epoch": t+1,
                        "best_psnr": best_psnr,
                        "best_ssim": best_ssim,
                        "best_loss": best_loss,
                        "best_lpips": best_lpips,
                        "best_epoch": best_epoch,
                        "best_delta_lab": best_delta_lab,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    }

                    if best_psnr == test_psnr:
                        # model save
                        path = "./model/{}_best.pth".format(self.log[:-4])
                        torch.save(save_dict, path)


                    print('Best test loss %f, PSNR %f SSIM %f LPIPS %f Delta_LAB %f' % (best_loss, best_psnr, best_ssim, best_lpips, best_delta_lab))
                    self.f.write('Best test loss %f, PSNR %f SSIM %f LPIPS %f Delta_LAB %f\n' % (best_loss, best_psnr, best_ssim, best_lpips, best_delta_lab))

                    print('Best test loss2 %f, PSNR %f SSIM %f LPIPS %f Delta_LAB %f' % (best_loss2, best_psnr2, best_ssim2, best_lpips2, best_delta_lab2))
                    self.f.write('Best test loss2 %f, PSNR %f SSIM %f LPIPS %f Delta_LAB %f\n' % (best_loss2, best_psnr2, best_ssim2, best_lpips2, best_delta_lab2))

                    print('Best test loss3 %f, PSNR %f SSIM %f LPIPS %f Delta_LAB %f' % (best_loss3, best_psnr3, best_ssim3, best_lpips3, best_delta_lab3))
                    self.f.write('Best test loss3 %f, PSNR %f SSIM %f LPIPS %f Delta_LAB %f\n' % (best_loss3, best_psnr3, best_ssim3, best_lpips3, best_delta_lab3))
                    path = "./model/{}_latest.pth".format(self.log[:-4])
                    torch.save(save_dict, path)



        print('Best test loss %f, PSNR %f SSIM %f' % (best_loss, best_psnr, best_ssim))
        self.f.write('Best test loss %f, PSNR %f SSIM %f\n' % (best_loss, best_psnr, best_ssim))

        print('Best test loss2 %f, PSNR %f SSIM %f' % (best_loss2, best_psnr2, best_ssim2))
        self.f.write('Best test loss2 %f, PSNR %f SSIM %f\n' % (best_loss2, best_psnr2, best_ssim2))
        return best_loss, best_psnr, best_ssim, best_lpips

    def test(self, data):
        """Testing"""
        self.model.train(False)
        device = self.device
        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        epoch_lpips = 0
        epoch_delta_lab = 0
        n = 0
        if self.saveimg != 0:
            img_path = "./model/{}".format(self.log[:-4])
            if not os.path.exists(img_path):
                os.makedirs(img_path)

        if self.config.write_text == 1 and self.config.rank == 0:
            f1 = open("transform_params.txt", 'w')
            f2 = open("offset_param.txt", 'w')
            f3 = open("hyper_params.txt", 'w')
            np.set_printoptions(linewidth=np.inf)
            np.set_printoptions(suppress=True)
            np.set_printoptions(precision=5)

        for img, label, index, img_idx in data:
            N, C, H, W = img.shape
            temp = [i / (self.control_point + 1) for i in range(self.control_point + 2)]
            color_position = torch.tensor(temp)
            color_position = color_position.unsqueeze(0).unsqueeze(1)
            color_position = color_position.repeat(N, self.config.feature_num, 1)
            color_position = torch.tensor(color_position.cuda(device))
            # Data.
            img = torch.tensor(img.cuda(device))
            index = torch.tensor(index.cuda(device))
            label = torch.tensor(label.cuda(device))

            n = n + 1
            if n % int(data.__len__() / 10) == 0:
                sys.stdout.write('\rTest {}/{} '.format(n, data.__len__()))
                self.f.write('Test {}/{}\n'.format(n, data.__len__()))

            if self.modeln == 30 or self.modeln == 31:
                pred, params = self.model(img, index, color_position)
            else:
                if self.config.write_text == 0:
                    pred = self.model(img, index, color_position)
                else:
                    pred, transform_params, offset_param, hyper_params = self.model(img, index, color_position)
                    transform_params = transform_params.reshape(self.config.feature_num * 3)
                    #offset_param = offset_param.reshape(self.config.feature_num * (self.config.control_point + 2))
                    offset_param = offset_param.reshape(self.config.feature_num, (self.config.control_point + 2))
                    hyper_params = hyper_params.reshape(self.config.feature_num * 3)
                    
                    transform_params = transform_params.cpu().detach().numpy()
                    offset_param = offset_param.cpu().detach().numpy()
                    hyper_params = hyper_params.cpu().detach().numpy()
                    #hypers.append(hyper_params.cpu().detach().numpy())
                    transform_params = np.array2string(transform_params)[1:-1]
                    
                    hyper_params = np.array2string(hyper_params)[1:-1]
                    if self.config.rank == 0:
                        f1.write("{}\n".format(transform_params))
                        for i in range(0, self.config.feature_num ):
                            offset_param_temp = np.array2string(offset_param[i,:])
                            offset_param_temp = np.array2string(offset_param[i,:])[1:-1]
                            f2.write("{}\n".format(offset_param_temp))
                        f3.write("{}\n".format(hyper_params))
                    #np.savetxt('my_file.txt', torch.Tensor([3,4,5,6]).numpy())
            #pred = self.model(img, index, color_position)
                
            if self.vgg_loss == 0:
                loss = self.l1_loss(pred, label)
            else:
                loss = self.l1_loss(pred, label) + self.vgg * self.vgg_criterion(pred, label)
            epoch_loss = epoch_loss + loss.detach().cpu().numpy()

            if self.norm == 1:
                pred = 0.5 * (pred + 1.0)
                label = 0.5 * (label + 1.0)
                
            pred = torch.clamp(pred, 0, 1)
            
            psnr = self.PSNR(pred, label)
            epoch_psnr = epoch_psnr + psnr.detach().cpu().numpy()

            ssim = structural_similarity(pred, label)
            epoch_ssim = epoch_ssim + ssim.detach().cpu().numpy()


            pred_lpips = pred.detach() * 2.0 - 1.0
            label_lpips = label.detach() * 2.0 - 1.0
            lpips = torch.mean(self.lpips_fn(pred_lpips, label_lpips).squeeze())

            epoch_lpips = epoch_lpips + lpips.detach().cpu().numpy()

            delta_lab = calculate_delta_lab(pred, label)
            epoch_delta_lab = epoch_delta_lab + delta_lab.detach().cpu().numpy()

            #save_img
            if self.saveimg != 0:
                img_path2 = "{}/{:04d}.png".format(img_path, img_idx[0])
                save_image(pred, img_path2)

        if self.config.write_text == 1 and self.config.rank == 0:
            f1.close()
            f2.close()
            f3.close()
        #print('\n{} images\n'.format(n))
        epoch_loss = epoch_loss / n
        epoch_psnr = epoch_psnr / n
        epoch_ssim = epoch_ssim / n
        epoch_lpips = epoch_lpips / n
        epoch_delta_lab = epoch_delta_lab / n
        
        self.model.train(True)
        return epoch_loss, epoch_psnr, epoch_ssim, epoch_lpips, epoch_delta_lab