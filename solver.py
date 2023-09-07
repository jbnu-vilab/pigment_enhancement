import os


import data_loader
import time
import sys
import torch.nn as nn
import kornia
import math
import torch
from scheduler import WarmupCosineSchedule
from torchvision.utils import save_image
from vggloss import VGGPerceptualLoss, VGGContrastiveLoss
import lpips
import models_proposed


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



def structural_similarity(outputs, targets):
    outputs = torch.clamp(outputs, 0, 1)
    targets = torch.clamp(targets, 0, 1)

    ssim = kornia.metrics.ssim(outputs, targets, window_size=11, max_val=1.0)
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
        self.log = config.log
        self.dataset = config.dataset
        self.saveimg = config.saveimg
        self.test_step = config.test_step
        self.vgg_loss = config.vgg_loss
        self.tv = config.total_variation
        self.vgg =config.vgg
        self.norm = config.norm
        self.m = config.m
        self.iter_num = config.iter_num
        self.weight_mode = config.weight_mode
        self.style_loss = config.style_loss

        if config.loss == 'l1':
            self.l1_loss = torch.nn.L1Loss().cuda()
        elif config.loss == 'l2':
            self.l1_loss = torch.nn.MSELoss().cuda()

        self.lpips_fn = lpips.LPIPS().cuda()


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        if config.m == 23:
            self.model = models_proposed.DCPNet23(config).cuda()
        elif config.m == 24:
            self.model = models_proposed.DCPNet24(config).cuda()
        elif config.m == 25:
            self.model = models_proposed.DCPNet25(config).cuda()

       

        self.PSNR = PSNR().cuda()
        self.PSNR.training = False
        self.lr = config.lr
        self.weight_decay = config.weight_decay

        train_loader = data_loader.DataLoader(config.dataset, path, config=config, batch_size=config.batch_size, istrain=True, num_workers=config.num_workers)
        test_loader = data_loader.DataLoader(config.dataset, path, config=config, batch_size=1, istrain=False)

        batch_step_num = math.ceil(train_loader.data.__len__() / config.batch_size)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if config.scheduler == 'cos_warmup':
            self.scheduler = WarmupCosineSchedule(self.optimizer, warmup_steps=math.ceil(batch_step_num * config.warmup_step), t_total=batch_step_num * config.epochs, cycles=0.5)

        if config.resume == 1: # resume to the latest epoch
            checkpoint = torch.load('./model/{}_latest.pth'.format(self.log[:-4]))

            self.model.load_state_dict(checkpoint["model"], strict=False)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_epoch = checkpoint["epoch"]
            self.best_psnr = checkpoint["best_psnr"]
            self.best_loss = checkpoint["best_loss"]
            self.best_ssim = checkpoint["best_ssim"]
            self.best_lpips = checkpoint["best_lpips"]
            print(self.start_epoch, self.best_psnr, self.best_loss, self.best_ssim, self.best_lpips)
        elif config.resume == 2: # resume to the best epoch
            checkpoint = torch.load('./model/{}_best.pth'.format(self.log[:-4]))

            self.model.load_state_dict(checkpoint["model"], strict=False)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_epoch = checkpoint["epoch"]

        else:
            self.start_epoch = 0
            self.best_psnr = 0
            self.best_loss = 100
            self.best_ssim = 0
            self.best_lpips = 100

        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

        self.f = config.f
        self.control_point = config.control_point

    def train(self):
        best_psnr = self.best_psnr
        best_loss = self.best_loss
        best_ssim = self.best_ssim
        best_lpips = self.best_lpips

        best_psnr2 = self.best_psnr
        best_loss2 = self.best_loss
        best_ssim2 = self.best_ssim
        best_lpips2 = self.best_lpips

        best_psnr3 = self.best_psnr
        best_loss3 = self.best_loss
        best_ssim3 = self.best_ssim
        best_lpips3 = self.best_lpips

        for t in range(self.start_epoch, self.epochs):
            epoch_loss = 0
            epoch_psnr = 0
            epoch_ssim = 0
            epoch_lpips = 0
            i = 0

            start = time.time()
            for img, label, index in self.train_data:
                i = i+1
                N, C, H, W = img.shape
                temp = [i / (self.control_point+1) for i in range(self.control_point+2)]
                color_position = torch.tensor(temp)
                color_position = color_position.unsqueeze(0).unsqueeze(1)
                color_position = color_position.repeat(N, self.config.feature_num, 1)
                color_position = torch.tensor(color_position.cuda())

                img = torch.tensor(img.cuda())
                index = torch.tensor(index.cuda())
                label = torch.tensor(label.cuda())
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



                pred = self.model(img, index, color_position)
                if self.vgg_loss == 0:
                    loss = self.l1_loss(pred, label)
                else:
                    loss = self.l1_loss(pred, label) + self.vgg * self.vgg_criterion(pred, label)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss = epoch_loss + loss.detach().cpu().numpy()
                
                if self.norm == 1:
                    pred = 0.5 * (pred + 1.0)
                    label = 0.5 * (label + 1.0)
                psnr = self.PSNR(pred, label)

                ssim = structural_similarity(pred, label)

                pred_lpips = pred.detach() * 2.0 - 1.0
                label_lpips = label.detach() * 2.0 - 1.0
                lpips = torch.mean(self.lpips_fn(pred_lpips, label_lpips).squeeze())

                epoch_psnr = epoch_psnr + psnr.detach().cpu().numpy()
                epoch_ssim = epoch_ssim + ssim.detach().cpu().numpy()
                epoch_lpips = epoch_lpips + lpips.detach().cpu().numpy()

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
            print('train loss %f, PSNR %f SSIM %f LPIPS %f' % (epoch_loss, epoch_psnr, epoch_ssim, epoch_lpips))
            self.f.write('train loss %f, PSNR %f SSIM %f LPIPS %f\n' % (epoch_loss, epoch_psnr, epoch_ssim, epoch_lpips))
            if (t+1) % self.test_step == 0:
                with torch.no_grad():
                    test_loss, test_psnr, test_ssim, test_lpips = self.test(self.test_data)
                    print('test loss %f, PSNR %f SSIM %f LPIPS %f' % (test_loss, test_psnr, test_ssim, test_lpips))
                    self.f.write('test loss %f, PSNR %f SSIM %f LPIPS %f\n' % (test_loss, test_psnr, test_ssim, test_lpips))

                    if best_psnr < test_psnr:
                        best_psnr = test_psnr
                        best_loss = test_loss
                        best_ssim = test_ssim
                        best_lpips = test_lpips

                    if best_ssim2 < test_ssim:
                        best_psnr2 = test_psnr
                        best_loss2 = test_loss
                        best_ssim2 = test_ssim
                        best_lpips2 = test_lpips

                    if best_lpips3 > test_lpips:
                        best_psnr3 = test_psnr
                        best_loss3 = test_loss
                        best_ssim3 = test_ssim
                        best_lpips3 = test_lpips



                    save_dict = {
                        "epoch": t+1,
                        "best_psnr": best_psnr,
                        "best_ssim": best_ssim,
                        "best_loss": best_loss,
                        "best_lpips": best_lpips,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    }

                    if best_psnr == test_psnr:
                        # model save
                        path = "./model/{}_best.pth".format(self.log[:-4])
                        torch.save(save_dict, path)


                    print('Best test loss %f, PSNR %f SSIM %f LPIPS %f' % (best_loss, best_psnr, best_ssim, best_lpips))
                    self.f.write('Best test loss %f, PSNR %f SSIM %f LPIPS %f\n' % (best_loss, best_psnr, best_ssim, best_lpips))

                    print('Best test loss2 %f, PSNR %f SSIM %f LPIPS %f' % (best_loss2, best_psnr2, best_ssim2, best_lpips2))
                    self.f.write('Best test loss2 %f, PSNR %f SSIM %f LPIPS %f\n' % (best_loss2, best_psnr2, best_ssim2, best_lpips2))

                    print('Best test loss3 %f, PSNR %f SSIM %f LPIPS %f' % (best_loss3, best_psnr3, best_ssim3, best_lpips3))
                    self.f.write('Best test loss3 %f, PSNR %f SSIM %f LPIPS %f\n' % (best_loss3, best_psnr3, best_ssim3, best_lpips3))
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

        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        epoch_lpips = 0
        n = 0
        if self.saveimg != 0:
            img_path = "./model/{}".format(self.log[:-4])
            if not os.path.exists(img_path):
                os.makedirs(img_path)


        for img, label, index in data:
            N, C, H, W = img.shape
            temp = [i / (self.control_point + 1) for i in range(self.control_point + 2)]
            color_position = torch.tensor(temp)
            color_position = color_position.unsqueeze(0).unsqueeze(1)
            color_position = color_position.repeat(N, self.config.feature_num, 1)
            color_position = torch.tensor(color_position.cuda())
            # Data.
            img = torch.tensor(img.cuda())
            index = torch.tensor(index.cuda())
            label = torch.tensor(label.cuda())

            n = n + 1
            if n % int(data.__len__() / 10) == 0:
                sys.stdout.write('\rTest {}/{} '.format(n, data.__len__()))
                self.f.write('Test {}/{}\n'.format(n, data.__len__()))




            pred = self.model(img, index, color_position)

            if self.vgg_loss == 0:
                loss = self.l1_loss(pred, label)
            else:
                loss = self.l1_loss(pred, label) + self.vgg * self.vgg_criterion(pred, label)
            epoch_loss = epoch_loss + loss.detach().cpu().numpy()

            if self.norm == 1:
                pred = 0.5 * (pred + 1.0)
                label = 0.5 * (label + 1.0)
                
            psnr = self.PSNR(pred, label)
            epoch_psnr = epoch_psnr + psnr.detach().cpu().numpy()

            ssim = structural_similarity(pred, label)
            epoch_ssim = epoch_ssim + ssim.detach().cpu().numpy()


            pred_lpips = pred.detach() * 2.0 - 1.0
            label_lpips = label.detach() * 2.0 - 1.0
            lpips = torch.mean(self.lpips_fn(pred_lpips, label_lpips).squeeze())

            epoch_lpips = epoch_lpips + lpips.detach().cpu().numpy()

            #save_img
            if self.saveimg != 0:

                img_path2 = "{}/{:04d}.png".format(img_path, n)
                save_image(pred, img_path2)

        print('\n')
        epoch_loss = epoch_loss / n
        epoch_psnr = epoch_psnr / n
        epoch_ssim = epoch_ssim / n
        epoch_lpips = epoch_lpips / n
        self.model.train(True)
        return epoch_loss, epoch_psnr, epoch_ssim, epoch_lpips