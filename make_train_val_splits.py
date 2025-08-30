import argparse
import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset
from PIL import Image
import pandas as pd
## Progress bar
from tqdm.auto import tqdm
import numpy as np
import random
import shutil
from functools import partial

def create_labels_dir(dir, labels):
    for label in labels:
        save_dir_label = dir + str(label) + '/'
        os.makedirs(save_dir_label, exist_ok=True)

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    input_dir = os.path.join(args.input_dir, str(args.target_image_size) + '/')
    output_dir = os.path.join(args.output_dir, str(args.target_image_size) + '/')
    save_dir_train = output_dir + 'train/'
    os.makedirs(save_dir_train, exist_ok=True)
    save_dir_val = output_dir + 'val/'
    os.makedirs(save_dir_val, exist_ok=True)
    save_dir_test = output_dir + 'test/'
    os.makedirs(save_dir_test, exist_ok=True)
    
    if args.select_method == 'train':
        with os.scandir(input_dir) as env_dir:      # env_dir is directory of per-label sub-directories
            do_test = env_dir.name != args.test_domain
            with os.scandir(env_dir) as lab_dir:    # lab_dir is a label sub-directory
                if lab_dir.is_dir():
                    with os.scandir(lab_dir) as fs:     # fs are the images of a label
                        files = [f for f in fs if f.is_file()]
                        label = lab_dir.name

                        if not do_test:
                            num_files = len(files)
                            f_idx = np.random.permutation(num_files)
                            train_num = int(num_files*args.train_split)
                            train_idx = f_idx[:train_num]
                            val_idx = f_idx[train_num:]

                            output_lab_dir = os.path.join(save_dir_train, label + '/')
                            os.makedirs(output_lab_dir, exist_ok=True)
                            for fp in [files[i] for i in train_idx]:
                                shutil.copyfile(fp, output_lab_dir)
                            output_lab_dir = os.path.join(save_dir_val, label + '/')
                            os.makedirs(output_lab_dir, exist_ok=True)
                            for fp in [files[i] for i in val_idx]:
                                shutil.copyfile(fp, output_lab_dir)
                        else:
                            output_lab_dir = os.path.join(save_dir_test, label + '/')
                            os.makedirs(output_lab_dir, exist_ok=True)
                            for fp in [files[i] for i in train_idx]:
                                shutil.copyfile(fp, output_lab_dir)
    elif args.select_method == 'loo':
        with os.scandir(input_dir) as env_dir:      # env_dir is directory of per-label sub-directories
            if env_dir.is_dir():
                if env_dir == args.val_dir:
                    output_task_dir = save_dir_val
                elif env_dir == args.test_dir:
                    output_task_dir = save_dir_test
                else:
                    output_task_dir = save_dir_train
                shutil.copytree(env_dir, output_task_dir, dir_exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Global args
    parser.add_argument('--input_dir', type=str, default="./data/DataSets/terra_incognita/JPEG")
    parser.add_argument('--output_dir', type=str, default="./data/DataSets/terra_incognita/JPEG")
    parser.add_argument('--target_image_size', type=int, default=224)
    parser.add_argument('--test_domain', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--domain_names', type=str, nargs='+', required=True, help='Cannot be last before selection method')

    subparsers = parser.add_subparsers(dest='select_method', required=True)

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--train_split', type=partial(bounded_type, min_val=0.0, max_val=1.0, cast_type=float), required=True)

    loo_parser = subparsers.add_parser('loo')
    loo_parser.add_argument('--val_domain', type=str, required=True)

    args = parser.parse_args()
    
    main(args)


