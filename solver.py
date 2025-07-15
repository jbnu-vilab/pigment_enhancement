import os

import data_loader
import sys
import torch.nn as nn
import kornia
import math
import torch
from scheduler import WarmupCosineSchedule
from torchvision.utils import save_image
import lpips
import models_proposed

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




class solver_IE(object):
    """Solver for training and testing"""
    def __init__(self, config, path):
        self.epochs = config.epochs
        self.log = config.logs
        self.dataset = config.dataset
        self.saveimg = config.saveimg
        self.test_step = config.test_step
        self.model = config.model
        self.iter_num = config.iter_num
        self.weight_mode = config.weight_mode
        self.style_loss = config.style_loss
        self.parallel = config.parallel
        self.modeln = config.model

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.l1_loss = torch.nn.L1Loss().cuda(device)
        self.lpips_fn = lpips.LPIPS().cuda(device)
        
        self.config = config
        self.model = models_proposed.PigNet(config).cuda()
        self.PSNR = PSNR().cuda(device)
        self.PSNR.training = False
        self.lr = config.lr
        self.lrratio = config.lrratio
        self.weight_decay = config.weight_decay

        train_loader = data_loader.DataLoader(config.dataset, path, config=config, batch_size=config.batch_size, istrain=True, num_workers=config.num_workers)
        test_loader = data_loader.DataLoader(config.dataset, path, config=config, batch_size=1, istrain=False)

        batch_step_num = math.ceil(train_loader.data.__len__() / config.batch_size)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.scheduler = WarmupCosineSchedule(self.optimizer, warmup_steps=math.ceil(batch_step_num * config.warmup_step), t_total=batch_step_num * config.epochs, cycles=0.5)

        if config.resume == 1: # resume to the latest epoch
            checkpoint = torch.load('./model/{}_latest.pth'.format(self.log[:-4]))

            self.model.load_state_dict(checkpoint["model"], strict=False)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_epoch = checkpoint["epoch"]
            self.best_psnr = checkpoint["best_psnr"]
            self.best_loss = checkpoint["best_loss"]
            self.best_lpips = checkpoint["best_lpips"]
            self.best_epoch = checkpoint["best_epoch"]
            if 'best_delta_lab' in checkpoint.keys():
                self.best_delta_lab = checkpoint["best_delta_lab"]
            else:
                self.best_delta_lab = 100
            print(self.start_epoch, self.best_psnr, self.best_loss, self.best_lpips)
        elif config.resume == 2: # resume to the best epoch
            checkpoint = torch.load('./model/{}_best.pth'.format(self.log[:-4]))
            self.model.load_state_dict(checkpoint["model"], strict=False)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_epoch = checkpoint["epoch"]
            self.best_psnr = checkpoint["best_psnr"]
            self.best_loss = checkpoint["best_loss"]
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
        best_lpips = self.best_lpips
        best_delta_lab = self.best_delta_lab

        best_psnr2 = self.best_psnr
        best_loss2 = self.best_loss
        best_lpips2 = self.best_lpips
        best_delta_lab2 = self.best_delta_lab


        best_epoch = self.best_epoch
        device = self.device

        for t in range(self.start_epoch, self.epochs):

            epoch_loss = 0
            epoch_psnr = 0
            epoch_lpips = 0
            epoch_delta_lab = 0
            i = 0

            for img, label, index, img_idx in self.train_data:
                i = i+1
                N, C, H, W = img.shape
                temp = [i / (self.control_point+1) for i in range(self.control_point+2)]
                color_position = torch.tensor(temp)
                color_position = color_position.unsqueeze(0).unsqueeze(1)
                color_position = color_position.repeat(N, self.config.feature_num, 1)

                color_position = color_position.cuda(device)

                img = img.cuda(device)
                index = index.cuda(device)
                label = label.cuda(device)
                self.optimizer.zero_grad()


                pred = self.model(img, index, color_position)
                loss = self.l1_loss(pred, label)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                with torch.no_grad():
                    epoch_loss = epoch_loss + loss.detach().cpu().numpy()

                    psnr = self.PSNR(pred, label)

                    delta_lab = calculate_delta_lab(pred, label)

                    pred_lpips = pred.detach() * 2.0 - 1.0
                    label_lpips = label.detach() * 2.0 - 1.0
                    lpips = torch.mean(self.lpips_fn(pred_lpips, label_lpips).squeeze())

                    epoch_psnr = epoch_psnr + psnr.detach().cpu().numpy()
                    
                    epoch_lpips = epoch_lpips + lpips.detach().cpu().numpy()
                    epoch_delta_lab = epoch_delta_lab + delta_lab.detach().cpu().numpy()

                    if i % 20 == 1:
                        sys.stdout.write('\rEpoch {}: {}/{}, loss: {}'.format(t + 1, i, self.train_data.__len__(), loss))
                        self.f.write('Epoch {}: {}/{}, loss: {}\n'.format(t + 1, i, self.train_data.__len__(), loss))


            epoch_loss = epoch_loss / i
            epoch_psnr = epoch_psnr / i
            epoch_lpips = epoch_lpips / i
            epoch_delta_lab = epoch_delta_lab / i
            print('\ntrain loss %f, PSNR %f, LPIPS %f, Delta_LAB %f' % (epoch_loss, epoch_psnr, epoch_lpips, epoch_delta_lab))
            if (t+1) % self.test_step == 0:
                with torch.no_grad():
                    test_loss, test_psnr, test_lpips, test_delta_lab = self.test(self.test_data)
                    print('test loss %f, PSNR %f, LPIPS %f, Delta_LAB %f' % (test_loss, test_psnr, test_lpips, test_delta_lab))
                    if best_psnr < test_psnr:
                        best_psnr = test_psnr
                        best_loss = test_loss
                        best_lpips = test_lpips
                        best_delta_lab = test_delta_lab
                        best_epoch = t

                    if best_delta_lab2 > test_delta_lab:
                        best_psnr2 = test_psnr
                        best_loss2 = test_loss
                        best_lpips2 = test_lpips
                        best_delta_lab2 = test_delta_lab

                    save_dict = {
                        "epoch": t+1,
                        "best_psnr": best_psnr,
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

                    print('Best test loss %f, PSNR %f LPIPS %f Delta_LAB %f' % (best_loss, best_psnr, best_lpips, best_delta_lab))

                    print('Best test loss2 %f, PSNR %f LPIPS %f Delta_LAB %f' % (best_loss2, best_psnr2, best_lpips2, best_delta_lab2))

                    path = "./model/{}_latest.pth".format(self.log[:-4])
                    torch.save(save_dict, path)



        print('Best test loss %f, PSNR %f LPIPS %f Delta_LAB %f' % (best_loss, best_psnr, best_lpips, best_delta_lab))

        print('Best test loss2 %f, PSNR %f LPIPS %f Delta_LAB %f' % (best_loss2, best_psnr2, best_lpips2, best_delta_lab2))
        return best_loss, best_psnr, best_lpips

    def test(self, data):
        """Testing"""
        self.model.train(False)
        device = self.device
        epoch_loss = 0
        epoch_psnr = 0
        epoch_lpips = 0
        epoch_delta_lab = 0
        n = 0
        if self.saveimg != 0:
            img_path = "./model/{}".format(self.log[:-4])
            if not os.path.exists(img_path):
                os.makedirs(img_path)


        with torch.no_grad():
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


                pred = self.model(img, index, color_position)
                
                loss = self.l1_loss(pred, label)
                epoch_loss = epoch_loss + loss.detach().cpu().numpy()


                pred = torch.clamp(pred, 0, 1)
                
                psnr = self.PSNR(pred, label)
                epoch_psnr = epoch_psnr + psnr.detach().cpu().numpy()

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


            epoch_loss = epoch_loss / n
            epoch_psnr = epoch_psnr / n
            epoch_lpips = epoch_lpips / n
            epoch_delta_lab = epoch_delta_lab / n
        
        self.model.train(True)
        return epoch_loss, epoch_psnr, epoch_lpips, epoch_delta_lab