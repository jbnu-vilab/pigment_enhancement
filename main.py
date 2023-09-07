import os
import argparse
import random
import numpy as np
import torch

from solver import solver_IE


# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    folder_path = {
        'adobe5k': '../DB/Enhancement_DB/Adobe5k_480p_train_test/',
        'adobe5k2': '../DB/Enhancement_DB/Adobe5k_480p_train_test/',
        'LOL': '../DB/Enhancement_DB/LOLdataset/',
        'uieb': '../DB/Enhancement_DB/UIEB_HU/',
        'euvp': '../DB/Enhancement_DB/euvp/',
        'hdr': '../DB/Enhancement_DB/hdrplus/',
        'ppr10ka': '../DB/Enhancement_DB/train_val_images_tif_360p/',
        'ppr10kb': '../DB/Enhancement_DB/train_val_images_tif_360p/',
        'ppr10kc': '../DB/Enhancement_DB/train_val_images_tif_360p/'
        #'adobe5k': '../DB/Enhancement_DB/Adobe5k_train_test/',
    }
    if os.path.exists('log') == False:
        os.mkdir('log')
    if os.path.exists('model') == False:
        os.mkdir('model')


    if config.seed == 0:
        pass
    else:
        print('we are using the seed = {}'.format(config.seed))
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

    #torch.backends.cudnn.enabled = False
    torch.cuda.set_device(0)

    if config.test == False:

        config.f = open("log/{}".format(config.log), 'w')
        print('Training and testing on %s dataset...' % (config.dataset))
        config.f.write('Training and testing on %s dataset...')
        solver = solver_IE(config, folder_path[config.dataset])
        best_loss, best_psnr, best_ssim, best_lpips = solver.train()
    else:
        #config.f = open("{}".format(config.log), 'w')
        config.f = open("log/{}".format(config.log), 'a')
        print('Training and testing on %s dataset...' % (config.dataset))
        config.f.write('Training and testing on %s dataset...' % (config.dataset))
        solver = solver_IE(config, folder_path[config.dataset])
        best_loss, best_psnr, best_ssim, best_lpips = solver.test(solver.test_data)
        print("loss: {}, psnr: {}, ssim: {}, lpips: {}".format(best_loss, best_psnr, best_ssim, best_lpips))
        config.f.write("loss: {}, psnr: {}, ssim: {}, lpips: {}".format(best_loss, best_psnr, best_ssim, best_lpips))


    config.f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='adobe5k',
                        help='Support datasets: adobe5k / LOL')
    parser.add_argument('--lr', dest='lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=256, help='Crop size for training & testing image patches')
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scheduler", dest='scheduler', type=str, default='cos_warmup', help='cos_warmup')
    parser.add_argument("--log", dest='log', type=str, default='temp.txt', help='log file')
    parser.add_argument("--resume", dest='resume', type=int, default=0, help='resume') # 1 latest 2 best

    parser.add_argument('--warmup_step', dest='warmup_step', type=float, default=1.0, help='warmup step')
    parser.add_argument('--saveimg', dest='saveimg', type=int, default=1, help='image save')
    parser.add_argument("--gpu", dest='gpu', type=str, default='0', help='gpu index')
    parser.add_argument("--loss", dest='loss', type=str, default='l1', help='loss')
    parser.add_argument("--vgg_loss", dest='vgg_loss', type=int, default=1, help='loss')
    parser.add_argument("--test_step", type=int, default=1)

    parser.add_argument("--global", dest='global_m', type=int, default=1)
    parser.add_argument("--residual", dest='residual', type=int, default=1)

    parser.add_argument("--control_point", dest='control_point', type=int, default=14)


    parser.add_argument("--total_variation", dest='total_variation', type=float, default=0.0)

    parser.add_argument("--act", dest='act', type=str, default='silu')

    parser.add_argument("--scale", dest='scale', type=int, default=4)


    parser.add_argument("--use_param", dest='use_param', type=int, default=1)
    parser.add_argument("--num_workers", dest='num_workers', type=int, default=4)
    parser.add_argument("--trainable_gamma", dest='trainable_gamma', type=int, default=0)
    parser.add_argument("--trainable_offset", dest='trainable_offset', type=int, default=1)
    parser.add_argument("--offset_param", dest='offset_param', type=float, default=0.04)
    parser.add_argument("--offset_param2", dest='offset_param2', type=float, default=0.04)
    parser.add_argument("--gamma_param", dest='gamma_param', type=float, default=0.1)
    parser.add_argument("--jitter", type=int, default=0)
    parser.add_argument("--lpips", dest='lpips', type=int, default=1)


    parser.add_argument("--fix_mode", dest='fix_mode', type=int, default=0)

    parser.add_argument("--vgg", dest='vgg', type=float, default=0.1)
    parser.add_argument("--vgg_mode", dest='vgg_mode', type=int, default=0)
    parser.add_argument("--contrastive", dest='contrastive', type=int, default=0)

    parser.add_argument("--glo_mode", dest='glo_mode', type=int, default=0)
    parser.add_argument("--m", dest='m', type=int, default=0)
    
    parser.add_argument("--feature_num", dest='feature_num', type=int, default=64)
    parser.add_argument("--iter_num", dest='iter_num', type=int, default=2)
    parser.add_argument("--norm", dest='norm', type=int, default=0)
    parser.add_argument("--weight_mode", dest='weight_mode', type=int, default=0)
    
    parser.add_argument("--style_loss", dest='style_loss', type=float, default=0)
    parser.add_argument("--conv_num", dest='conv_num', type=int, default=1)
    
    parser.add_argument("--transformer", dest='transformer', type=int, default=1)
    parser.add_argument("--size", dest='size', type=int, default=448)
    
    parser.add_argument("--res_mode", dest='res_mode', type=int, default=0)

    parser.add_argument("--hyper", dest='hyper', type=int, default=0)
    parser.add_argument("--conv_mode", dest='conv_mode', type=int, default=3)

    parser.add_argument("--xoffset", dest='xoffset', type=int, default=0)

    parser.add_argument("--num_weight", dest='num_weight', type=int, default=1)

    parser.add_argument("--act_mode", dest='act_mode', type=str, default='sigmoid')
    parser.add_argument("--backbone", dest='backbone', type=str, default='res')

    config = parser.parse_args()
    main(config)

