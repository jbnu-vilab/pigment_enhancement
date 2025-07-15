import os
import argparse
from solver import solver_IE


def main(config):
    folder_path = {
        'adobe5k': '../DB/Enhancement_DB/Adobe5k_480p_train_test/',
        'ppr10ka': '../DB/Enhancement_DB/train_val_images_tif_360p/',
        'ppr10kb': '../DB/Enhancement_DB/train_val_images_tif_360p/',
        'ppr10kc': '../DB/Enhancement_DB/train_val_images_tif_360p/',
    }
    if os.path.exists('log') == False:
        os.mkdir('log')
    if os.path.exists('model') == False:
        os.mkdir('model')


    if config.test == False:
        print('Training and testing on %s dataset...' % (config.dataset))
        solver = solver_IE(config, folder_path[config.dataset])
        best_loss, best_psnr, best_lpips = solver.train()
    else:
        print('Training and testing on %s dataset...' % (config.dataset))
        solver = solver_IE(config, folder_path[config.dataset])
        best_loss, best_psnr, best_lpips, best_delta_lab = solver.test(solver.test_data)
        print("loss: {}, psnr: {}, lpips: {}, delta_lab: {}".format(best_loss, best_psnr, best_lpips, best_delta_lab))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='adobe5k',
                        help='Support datasets: adobe5k / LOL')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=400, help='Epochs for training')

    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model_name", dest='model_name', type=str, default='adobe5k_res5', help='model name')
    parser.add_argument("--resume", dest='resume', type=int, default=0, help='resume') # 1 latest / 2 best
    parser.add_argument('--warmup_step', dest='warmup_step', type=float, default=1.0, help='warmup step')
    parser.add_argument('--saveimg', dest='saveimg', type=int, default=0, help='image save')
    parser.add_argument("--gpu", dest='gpu', type=str, default='0', help='gpu index')
    parser.add_argument("--test_step", type=int, default=1)
    parser.add_argument("--control_point", dest='control_point', type=int, default=30)
    parser.add_argument("--num_workers", dest='num_workers', type=int, default=8)
    parser.add_argument("--lpips", dest='lpips', type=int, default=1)
    parser.add_argument("--feature_num", dest='feature_num', type=int, default=64)
    parser.add_argument("--iter_num", dest='iter_num', type=int, default=400)

    parser.add_argument("--backbone_type", dest='backbone_type', type=int, default=5)
    parser.add_argument("--loader_size", dest='loader_size', type=int, default=256)


    config = parser.parse_args()
    if config.dataset in ["ppr10ka", "ppr10kb", "ppr10kc"]:
        config.loader_size = 512

    main(config)

