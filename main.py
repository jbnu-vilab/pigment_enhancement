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

        config.f = open("log/{}".format(config.logs), 'w')
        print('Training and testing on %s dataset...' % (config.dataset))
        config.f.write('Training and testing on %s dataset...')
        solver = solver_IE(config, folder_path[config.dataset])
        best_loss, best_psnr, best_lpips = solver.train()

        
    else:
        config.f = open("log/{}".format(config.logs), 'a')
        print('Training and testing on %s dataset...' % (config.dataset))
        config.f.write('Training and testing on %s dataset...' % (config.dataset))
        solver = solver_IE(config, folder_path[config.dataset])
        best_loss, best_psnr, best_lpips, best_delta_lab = solver.test(solver.test_data)
        print("loss: {}, psnr: {}, lpips: {}, delta_lab: {}".format(best_loss, best_psnr, best_lpips, best_delta_lab))
        config.f.write("loss: {}, psnr: {}, lpips: {}, delta_lab: {}".format(best_loss, best_psnr, best_lpips, best_delta_lab))

    config.f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='adobe5k',
                        help='Support datasets: adobe5k / LOL')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=400, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=256, help='Crop size for training & testing image patches')
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--logs", dest='logs', type=str, default='temp.txt', help='log file')
    parser.add_argument("--resume", dest='resume', type=int, default=0, help='resume') # 1 latest 2 best

    parser.add_argument('--warmup_step', dest='warmup_step', type=float, default=1.0, help='warmup step')
    parser.add_argument('--saveimg', dest='saveimg', type=int, default=0, help='image save')
    parser.add_argument("--gpu", dest='gpu', type=str, default='0', help='gpu index')
    parser.add_argument("--test_step", type=int, default=1)

    parser.add_argument("--global", dest='global_m', type=int, default=1)

    parser.add_argument("--control_point", dest='control_point', type=int, default=30)


    parser.add_argument("--act", dest='act', type=str, default='silu')

    parser.add_argument("--scale", dest='scale', type=int, default=4)


    parser.add_argument("--use_param", dest='use_param', type=int, default=1)
    parser.add_argument("--num_workers", dest='num_workers', type=int, default=8)
    parser.add_argument("--trainable_gamma", dest='trainable_gamma', type=int, default=0)
    parser.add_argument("--trainable_offset", dest='trainable_offset', type=int, default=1)
    parser.add_argument("--offset_param", dest='offset_param', type=float, default=0.1)
    parser.add_argument("--offset_param2", dest='offset_param2', type=float, default=0)
    parser.add_argument("--gamma_param", dest='gamma_param', type=float, default=0.1)
    parser.add_argument("--lpips", dest='lpips', type=int, default=1)

    parser.add_argument("--seed_opt", dest='seed_opt', type=int, default=0)


    parser.add_argument("--feature_num", dest='feature_num', type=int, default=64)
    parser.add_argument("--iter_num", dest='iter_num', type=int, default=2)

    parser.add_argument("--conv_num", dest='conv_num', type=int, default=1)
    
    parser.add_argument("--transformer", dest='transformer', type=int, default=1)
    parser.add_argument("--size", dest='size', type=int, default=448)
    
    parser.add_argument("--res_mode", dest='res_mode', type=int, default=0)

    parser.add_argument("--num_weight", dest='num_weight', type=int, default=1)

    parser.add_argument("--backbone", dest='backbone', type=str, default='res')

    parser.add_argument("--res_num", dest='res_num', type=int, default=5)

    parser.add_argument("--res_size", dest='res_size', type=int, default=256)
    

    parser.add_argument("--loader_size", dest='loader_size', type=int, default=256)



    parser.add_argument("--upsample_mode", dest='upsample_mode', type=int, default=1)

    parser.add_argument("--trans_param", dest='trans_param', type=float, default=5.0)

    parser.add_argument("--learnable_trans_param", dest='learnable_trans_param', type=int, default=0)


    parser.add_argument("--fc_node", dest='fc_node', type=int, default=1024)
    parser.add_argument("--optimizer_debug", dest='optimizer_debug', type=int, default=0)
    

    parser.add_argument("--fc_node1", dest='fc_node1', type=int, default=128)
    parser.add_argument("--fc_node2", dest='fc_node2', type=int, default=128)
    
    config = parser.parse_args()

    main(config)

