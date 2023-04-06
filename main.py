import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
from model.runner import Runner


def get_args():
    parser = argparse.ArgumentParser()
    
    # optional
    parser.add_argument("-C", "--config", default='config.yaml')
    parser.add_argument("-c", "--checkpoint", help='checkpoint path')
    parser.add_argument("-g", "--cuda_devices", default='0')
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-l", "--localNet", type=str)
    parser.add_argument("-n", "--name_list", nargs='+', type=str)
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-L", "--lr", type=float)
    parser.add_argument("-w", "--weight_decay", type=float)
    parser.add_argument("-s", "--sigma", type=int)
    parser.add_argument("-x", "--mix_step", type=int)
    parser.add_argument("-u", "--use_background_channel", action='store_true')
    # required
    parser.add_argument("-r", "--run_name", type=str)
    parser.add_argument("-d", "--run_dir", type=str, default='./unet2d_loss/cephalometric')
    parser.add_argument(
        "-p", "--phase", choices=['train', 'validate', 'test'], default='test')
    return parser.parse_args()


if __name__ == "__main__":
    

    torch.autograd.set_detect_anomaly(True)
    
    args = get_args()
    Runner(args).run()
