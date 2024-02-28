# -*-coding:utf-8 -*-
'''
@File    :   main.py
@Time    :   2024/02/27 23:06:50
@Author  :   Madhu Korada 
@Version :   1.0
@Contact :   mkorada@cs.cmu.edu
@License :   (C)Copyright 2024-2025, Madhu Korada
@Desc    :   None
'''
# from fit_data import *
# from train_model import *
# from eval_model import *
import argparse
import fit_data, train_model, eval_model

def get_args_parser():
    parser = argparse.ArgumentParser('Main', add_help=False)
    parser.add_argument('--mode', default='fit', choices=['fit', 'train', 'eval'], type=str)
    
    # Fit parameters
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=100000, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int) # 1000 for eval
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--device', default='cuda', type=str) 
    
    # Training parameters
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--batch_size", default=32, type=int) # 1 for eval 
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--save_freq", default=2000, type=int)
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument('--load_feat', action='store_true') 
    parser.add_argument("--full_dataset", default=False, type=bool)
    parser.add_argument("--use_wandb", action="store_true") # Use wandb for logging
    parser.add_argument("--run_id", default=None, type=str)
    parser.add_argument("--use_pickle", action="store_true")
    
    # Evaluation parameters
    parser.add_argument('--vis_freq', default=50, type=int)
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    if args.mode == "fit":
        fit_data.train_model(args)
    elif args.mode == "train":
        train_model.train_model(args)
    elif args.mode == "eval":
        eval_model.evaluate_model(args)
    else:
        raise ValueError(f"Mode {args.mode} not supported")

    
