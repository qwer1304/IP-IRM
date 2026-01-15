import argparse
import os
import glob
import hashlib
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import random
import shutil
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset

from tqdm.auto import tqdm

import utils
from model import ModelResnet, SimSiam, MultiArmModel, create_mlp, Mask, MaskModule
import gradnorm as gn
import gc
from math import ceil, prod
import copy
import traceback
import sys
import time
import warnings
from collections import defaultdict
from typing import Union, List, Dict
from trainenv import train_env
from functools import partial

import utils_cluster

sys.modules['__main__'].FeatureQueue = utils.FeatureQueue

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, test_data_loader, args, num_classes, progress=False, prefix="Test:"):
    net.eval()
       
    total_top1, total_top5, total_num = 0.0, 0.0, 0
    with torch.no_grad():
        # loop test data to predict the label by weighted knn search
        bar_format = '{l_bar}{bar:' + str(args.bar) + '}{r_bar}' #{bar:-' + str(args.bar) + 'b}'
        
        if progress:
            test_bar = tqdm(test_data_loader,
                total=len(test_data_loader),
                ncols=args.ncols,               # total width available
                dynamic_ncols=False,            # disable autosizing
                bar_format=bar_format           # request bar width
            )
        else:
           test_bar = test_data_loader
    
        dataset = test_data_loader.dataset
        idcs    = list(range(len(dataset)))
        transform = dataset.transform
        target_transform = dataset.target_transform
    
        if args.extract_features:
            dataset.target_transform = None

        feature_list = []
        pred_labels_list = []
        pred_scores_list = []
        target_list = []
        target_raw_list = []
        # For macro-accuracy computation
        per_class_correct = torch.zeros(num_classes, dtype=torch.long, device=feature_bank[0].device)
        per_class_total   = torch.zeros(num_classes, dtype=torch.long, device=feature_bank[0].device)

        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            
            if transform is not None:
                data = transform(data)

            target_raw = target
            if args.extract_features and target_transform is not None:
                target = target_transform(target_raw).cuda(non_blocking=True)

            features = net.module.backbone(data)
            masked_features = net.module.mask(features)
            out = net.module.fc(masked_features)

            total_num += data.size(0)

            pred_labels = out.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            # predicted class is the top-1 label
            pred = pred_labels[:, 0]   # [B]

            # Loop-free update of per-class counts
            # For each class c: count how many predictions & targets match
            for cls in range(c):
                mask = (target == cls)
                if mask.any():
                    per_class_total[cls] += mask.sum()
                    per_class_correct[cls] += (pred[mask] == cls).sum()

            if progress:
                # Avoid division by zero in rare cases
                valid = per_class_total > 0
                macro_acc = (per_class_correct[valid].float() / per_class_total[valid].float()).mean().item()
                test_bar.set_description('KNN {} Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}% Macro-Acc:{:.2f}%'
                                         .format(prefix, epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100, macro_acc * 100))

            # compute output
            if args.extract_features:
                feature_list.append(feature)
                target_list.append(target)
                target_raw_list.append(target_raw)
                pred_labels_list.append(pred_labels)
                pred_scores_list.append(out)

        # end for data, _, target in test_bar

        # Avoid division by zero in rare cases
        valid = per_class_total > 0
        macro_acc = (per_class_correct[valid].float() / per_class_total[valid].float()).mean().item()

        if feature_list:
            feature = torch.cat(feature_list, dim=0)
            target = torch.cat(target_list, dim=0)
            target_raw = torch.cat(target_raw_list, dim=0)
            pred_labels = torch.cat(pred_labels_list, dim=0)
            pred_scores = torch.cat(pred_scores_list, dim=0)

            # Save to file
            prefix = "test" if "Test" in prefix else "val"
            directory = f'results-eqinv/{args.dataset}/{args.name}'
            fp = os.path.join(directory, f"{prefix}_features_dump.pt")       
            os.makedirs(os.path.dirname(fp), exist_ok=True)

            state = {
                'features':     feature,
                'labels':       target,
                'labels_raw':   target_raw,
                'pred_labels':  pred_labels,
                'pred_scores':  out,
                'model_epoch':  epoch,
                'n_classes':    args.class_num,
            }

            utils.atomic_save(state, False, filename=fp)
            print(f"Dumped features into {fp}")

    return total_top1 / total_num * 100, total_top5 / total_num * 100, macro_acc * 100
    
def load_checkpoint(path, model, model_momentum, optimizer, gradnorm_balancer, gradnorm_optimizer, device="cuda", classifier_not_needed=False):
    print("=> loading checkpoint '{}'".format(path))
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Restore training bookkeeping (if present)
    start_epoch = checkpoint.get("epoch", -1) + 1
    best_acc1 = checkpoint.get("best_acc1", 0.0)
    best_epoch = checkpoint.get("best_epoch", -1)
    updated_split = checkpoint.get("updated_split", None)
    updated_split_all = checkpoint.get("updated_split_all", None)
    ema = checkpoint.get("ema", None)


    # Restore main model
    msg_model = model.load_state_dict(checkpoint["state_dict"], strict=False)
    msg_model = msg_model._asdict()
    # Don't care about if classifier not present, if not needed
    if classifier_not_needed:
        msg_model['unexpected_keys'] = [k for k in msg_model['unexpected_keys'] if not k.startswith('module.arms.classifier')]

    # Restore momentum model if applicable
    queue = None
    if model_momentum is not None:
        if "state_dict_momentum" in checkpoint and checkpoint["state_dict_momentum"] is not None:
            msg_momentum = model_momentum.load_state_dict(
                checkpoint["state_dict_momentum"], strict=False
            )
            # Don't care about if classifier not present, if not needed
            if classifier_not_needed:
                msg_momentum = msg_momentum._asdict()
                msg_momentum['unexpected_keys'] = [k for k in msg_momentum['unexpected_keys'] if not k.startswith('module.arms.classifier')]
        else:
            msg_momentum = "no momentum encoder in checkpoint"

        if "queue" in checkpoint and checkpoint["queue"] is not None:
            queue = checkpoint["queue"]
        else:
            queue = None
    else:
        msg_momentum = "momentum encoder not used"
        queue = None
        
        if "state_dict_momentum" in checkpoint and checkpoint["state_dict_momentum"] is not None:
            msg_momentum = model_momentum.load_state_dict(
                checkpoint["state_dict_momentum"], strict=False
            )
        else:
            msg_momentum = "no momentum encoder in checkpoint"

        if "queue" in checkpoint and checkpoint["queue"] is not None:
            queue = checkpoint["queue"]
        else:
            queue = None    

    if (gradnorm_balancer is not None):
        if ("state_dict_gradnorm" in checkpoint) and (checkpoint["state_dict_gradnorm"] is not None):
            state_dict = checkpoint["state_dict_gradnorm"]
            if 'loss_keep' in state_dict["task_weights"]:
                state_dict["task_weights"]['loss_unsplit'] = state_dict["task_weights"].pop('loss_keep')
            new_parameters = gradnorm_balancer.load_state_dict(
                state_dict,
            )
            msg_gradnorm = f'new parameters: {new_parameters}'
        else:
            msg_gradnorm = "no gradnorm in checkpoint"
    else:
        msg_gradnorm = "gradnorm not used"

    # Restore optimizer (if available)
    if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:

        checkpoint["optimizer"]["param_groups"] = optimizer.param_groups  # keep current hparams
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Move optimizer tensors to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    if ("gradnorm_optimizer" in checkpoint) and (checkpoint["gradnorm_optimizer"] is not None):
        checkpoint["gradnorm_optimizer"]["param_groups"] = gradnorm_optimizer.param_groups  # keep current hparams
        gradnorm_optimizer.load_state_dict(checkpoint["gradnorm_optimizer"])
        # Move optimizer tensors to the correct device
        for state in gradnorm_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    # Restore RNG states (if present)
    rng_dict = checkpoint.get("rng_dict", None)
    if rng_dict is not None:
        rng_state = rng_dict.get("rng_state", None)
        if rng_state is not None:
            if rng_state.device != torch.device("cpu"):
                rng_state = rng_state.cpu()
            torch.set_rng_state(rng_state)

        cuda_rng_state = rng_dict.get("cuda_rng_state", None)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(
                [t.cpu() if t.device != torch.device("cpu") else t for t in cuda_rng_state]
            )

        numpy_rng_state = rng_dict.get("numpy_rng_state", None)
        if numpy_rng_state is not None:
            np.random.set_state(numpy_rng_state)

        python_rng_state = rng_dict.get("python_rng_state", None)
        if python_rng_state is not None:
            random.setstate(python_rng_state)

    # Report what was loaded
    print("\tmodel load: {}".format(msg_model))
    if model_momentum is not None:
        print("\tmomentum load: {}".format(msg_momentum))
    if queue is not None:
        print("\tqueue restored")
    if gradnorm_balancer is not None:
        print("\tgradnorm load: {}".format(msg_gradnorm))

    print("<= loaded checkpoint '{}' (epoch {})".format(path, checkpoint.get("epoch", -1)))

    return model, model_momentum, optimizer, queue, start_epoch, best_acc1, best_epoch, updated_split, updated_split_all, ema, gradnorm_balancer, gradnorm_optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EqInv')
    parser.add_argument('--ssl_type', default='MoCo', type=str, choices=['MoCo', 'SimSiam', 'MoCoSupCon'], help='SSL type')    
    parser.add_argument('--ssl_type_partition', default='MoCoSupCon', type=str, choices=['MoCo', 'SimSiam', 'MoCoSupCon', 'SimCLR'], help='SSL type for partition')    
    parser.add_argument('--penalty_type', default='IRM', type=str, choices=['IRM', 'VREx'], help='Penalty type')        
    parser.add_argument('--penalty_sigma', default=None, type=float, help='Noise level to inject into penalty')        
    parser.add_argument('--grad_rotate', default=None, type=float, nargs=2, help='rotate gradients')      
    parser.add_argument('--loss_unsplit_type', required=True, type=str, choices=['CE', 'CEweighted'], help='Loss unsplit type')    
    parser.add_argument('--irm_temp', default=0.5, type=float, help='irm loss temperature')

    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature in SSL softmax')
    parser.add_argument('--norandgray', action="store_true", default=False, help='skip rand gray transform')

    # Loaders parameters
    parser.add_argument('--dl_tr', default=[256, 4, 2, True, True], nargs=5, type=str,
                        action=utils.ParseMixed, types=[int, int, int, bool, bool],
                        metavar='DataLoader pars [batch_size, number_workers, prefetch_factor, persistent_workers, drop_last]',    
                        help='Training minimization DataLoader pars')
    parser.add_argument('--dl_u', default=[3096, 4, 2, 1], nargs=4, type=str,
                        action=utils.ParseMixed, types=[int, int, int, bool],
                        metavar='DataLoader pars [batch_size, number_workers, prefetch_factor, persistent_workers]', 
                        help='Training Maximization Image DataLoader pars')
    parser.add_argument('--dl_uo', default=[3096, 4, 2, 1], nargs=4, type=str,
                        action=utils.ParseMixed, types=[int, int, int, bool],
                        metavar='DataLoader pars [batch_size, number_workers, prefetch_factor, persistent_workers]', 
                        help='Training Maximization Features DataLoader pars')
    parser.add_argument('--dl_te', default=[3096, 4, 2, 1], nargs=4, type=str,
                        action=utils.ParseMixed, types=[int, int, int, bool],
                        metavar='DataLoader pars [batch_size, number_workers, prefetch_factor, persistent_workers]', help='Testing/Validation/Memory DataLoader pars')

    # Training driver parameters
    parser.add_argument('--micro_batch_size', default=32, type=int, help='batch size on gpu')
    parser.add_argument('--gradients_accumulation_batch_size', default=256, type=int, help='batch size of gradients accumulation')

    # MoCo queue parameters
    parser.add_argument('--queue_size', default=10000, type=int, help='momentum model queue size')
    parser.add_argument('--momentum', default=0.995, type=float, help='momentum model momentum')

    parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the dataset to train')

    parser.add_argument('--dataset', type=str, default='STL', choices=['STL', 'CIFAR10', 'CIFAR100', 'ImageNet'], help='experiment dataset')
    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('--name', type=str, default='None', help='experiment name')
    parser.add_argument('--baseline', action="store_true", default=False, help='SSL baseline?')
    parser.add_argument('--seed', type=int, default=0)

    #### model param ####
    parser.add_argument('--penalty_weight', default=1.0, type=float, help='invariance weight')
    parser.add_argument('--penalty_cont', default=1.0, type=float, help='cont loss weight')
    parser.add_argument('--unsplit_cont', action="store_true", default=False, help='unsplit original contrastive?')
    parser.add_argument('--penalty_unsplit_cont', default=1.0, type=float, help='unsplit loss weight')
    parser.add_argument('--retain_group', action="store_true", default=False, help='dummy')
    parser.add_argument('--Lscaler', default=1.0, type=float, help='Global scaler for losses gards')
    parser.add_argument('--penalty_iters', default=0, type=int, help='penalty weight start iteration')
    parser.add_argument('--increasing_weight', nargs=5, type=float, default=None, help='increasing penalty weight', 
            metavar='penalty_warmup, scale, speed, eps, debug')
    parser.add_argument('--env_num', default=2, type=int, help='number of environments in partition')
    parser.add_argument('--weight_env_eps', default=0., type=float, help='eps for per-env grad noise')

    parser.add_argument('--debug', action="store_true", default=False, help='debug?')
    parser.add_argument('--print_batch', action="store_true", default=False, help='print every batch')

    # image
    parser.add_argument('--image_size', type=int, default=224, help='image size')

    # domain/color in label
    parser.add_argument('--target_transform', type=str, default=None, help='a function definition to apply to target')
    parser.add_argument('--class_to_idx', type=str, default=None, help='a function definition to apply to class to obtain its index')
    parser.add_argument('--image_class', choices=['ImageNet', 'STL', 'CIFAR'], default='ImageNet', help='Image class, default=ImageNet')
    parser.add_argument('--class_num', default=1000, type=int, help='num of classes')

    parser.add_argument('--ncols', default=80, type=int, help='number of columns in terminal')
    parser.add_argument('--bar', default=50, type=int, help='length of progess bar')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrain_path', type=str, default=None, help='the path of pretrain model')
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--save_root', type=str, default='save', help='root dir for saving')
    parser.add_argument('--checkpoint_freq', default=3, type=int, metavar='N',
                    help='checkpoint epoch freqeuncy')
    parser.add_argument('--val_freq', default=3*3, type=int, metavar='N',
                    help='validation epoch freqeuncy')
    parser.add_argument('--test_freq', default=5*5, type=int, metavar='N',
                    help='test epoch freqeuncy')

    parser.add_argument('--evaluate', type=str, default=None, nargs="*", choices=['val', 'test'], help='only evaluate')
    parser.add_argument('--extract_features', action="store_true", help="extract features for post processing during evaluate")

    parser.add_argument('--opt', choices=['Adam', 'SGD'], default='Adam', help='Optimizer to use')
    parser.add_argument('--lr', default=0.001, type=float, help='LR')
    parser.add_argument('--SGD_momentum', default=0.9, type=float, help='LR')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay')
    parser.add_argument('--betas', default=[0.9, 0.999], type=float, nargs=2, help='Adam betas')

    parser.add_argument('--ema', type=str, default=None, choices=['reinit', 'retain'], help="adjust gradients w/ EMA")
    parser.add_argument('--gradnorm', action="store_true", help="use gradnorm")
    parser.add_argument('--gradnorm_epoch', default=0, type=int, help='gradnorm start epoch')
    parser.add_argument('--gradnorm_alpha', default=1.0, type=float, help='gradnorm alpha')
    parser.add_argument('--gradnorm_tau', default=None, nargs=2*3, type=str,
                        action=utils.ParseMixed, types=[str, float, str, float, str, float],
                        metavar='tau dictionary k-v pairs',    
                        help='loss divisors')
    parser.add_argument('--gradnorm_scalers', default=['loss_unsplit', 1.0, 'loss', 1.0, 'penalty', 1.0], nargs=2*3, type=str,
                        action=utils.ParseMixed, types=[str, float, str, float, str, float],
                        metavar='scalers dictionary k-v pairs',    
                        help='loss scalers when gradnorm inactive')
    parser.add_argument('--gradnorm_debug', type=str, default=None, choices=['gn', 'opt'], nargs='*', help="debug gradnorm", metavar='gn, optimize')
    parser.add_argument('--gradnorm_Gscaler', default=1.0, type=float, help='gradnorm loss scaler')
    parser.add_argument('--gradnorm_beta', default=1.0, type=float, help='gradnorm softplus')
    parser.add_argument('--gradnorm_avgG_detach_frac', default=0.0, type=float, help='gradnorm avg detach fraction')
    parser.add_argument('--gradnorm_loss_type', default='L1', type=str, choices=['L1', 'L2', "Huber"], help='gradnorm loss type')
    parser.add_argument('--gradnorm_lr', default=1e-3, type=float, help='gradnorm LR')
    parser.add_argument('--gradnorm_betas', default=[0.9, 0.999], type=float, nargs=2, help='gradnorm Adam betas')
    parser.add_argument('--gradnorm_weight_decay', default=1e-6, type=float, help='weight decay')
    parser.add_argument('--gradnorm_loss_lambda', default=0., type=float, help='gradnorm loss regularizer strength')
    parser.add_argument('--gradnorm_rescale_weights', action="store_true", help="rescale weights before starting")
    parser.add_argument('--gradnorm_huber_delta', default=1e-2, type=float, help='gradnorm Huber delta')

    parser.add_argument('--clamp_weights_for_progress', action="store_true", help="clamp loss' weights for progress")
    
    parser.add_argument('--adapt_bn', action="store_true", help="adapt BN layers")
    parser.add_argument('--featurizer_lr', type=float, default=0.0, help="featurizer LR")
    parser.add_argument('--projector_lr', type=float, default=0.0, help="projector LR")
    parser.add_argument('--predictor_lr', type=float, default=0.0, help="predictor LR")
    parser.add_argument('--bn_momentum', type=float, default=0.1, help="BN momentum")

    #### add mask
    parser.add_argument('--mask_nonlinearity', type=str, default='sigmoid', choices=['sigmoid', 'ident', 'gumbel'], help='type of non-linearity in mask')
    parser.add_argument('--opt_mask', action="store_true", default=False, help='optimize the mask')
    parser.add_argument('--gumbel_tau', type=float, default=1.0, help='tau for gumbel mask')
    parser.add_argument('--gumbel_soft', action="store_true", help='soft gumbel')
    parser.add_argument('--mask_sparsity', type=int, default=None, help='sparsity K s.t. # of hot masks <= K')
    parser.add_argument('--mask_sparsity_weight', type=float, default=0.0, help='weight of sparsity loss')

    # clustering
    parser.add_argument('--cluster_path', type=str, default=None, 
        help='path to cluster file. None means automatic creation ./misc/<name>/env_ref_set_<resumed|pretrained|default>')
    parser.add_argument('--only_cluster', action="store_true", help='only do clustering')
    parser.add_argument('--cluster_temp', type=float, default=0.1, help='temperature for clusteing') 
    parser.add_argument('--cluster_save_dist', action="store_true", help='save cluster distances in ./misc/<name>/env_ref_dist')
    parser.add_argument('--num_clusters', type=int, default=2, help='number of custer K') 
    parser.add_argument('--clusters_to_use', type=int, nargs='+', default=None, help='clusters to use out of K clusters') 

    # args parse
    args = parser.parse_args()
    args.gradnorm_tau = {args.gradnorm_tau[i]: args.gradnorm_tau[i+1] for i in range(0,len(args.gradnorm_tau),2)} if args.gradnorm_tau is not None else None
    args.gradnorm_scalers = {args.gradnorm_scalers[i]: args.gradnorm_scalers[i+1] for i in range(0,len(args.gradnorm_scalers),2)} if args.gradnorm_scalers is not None else None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.baseline:
        args.penalty_weight, args.penalty_cont = 0, 0
        
    assert ((args.penalty_weight > 0) or (args.penalty_cont > 0)      or  (args.penalty_unsplit_cont > 0)) or \
           ((args.penalty_cont == 0) and (args.penalty_unsplit_cont > 0) and (args.penalty_iters == 0))

    # seed
    utils.set_seed(args.seed)

    # warnings
    warnings.filterwarnings("always", category=UserWarning)
    
    feature_dim, temperature = args.feature_dim, args.temperature
    epochs  = args.epochs
    dl_tr, dl_te, dl_u, dl_uo = args.dl_tr, args.dl_te, args.dl_u, args.dl_uo
    target_transform = eval(args.target_transform) if args.target_transform is not None else None
    class_to_idx = eval(args.class_to_idx) if args.class_to_idx is not None else None
    image_class, image_size = args.image_class, args.image_size

    if not os.path.exists('results-eqinv/{}/{}'.format(args.dataset, args.name)):
        os.makedirs('results-eqinv/{}/{}'.format(args.dataset, args.name))
    log_file = 'results-eqinv/{}/{}/log.txt'.format(args.dataset, args.name)
    if not os.path.exists('{}/{}'.format(args.save_root, args.name)):
        os.makedirs('{}/{}'.format(args.save_root, args.name))

    # data prepare
    tr_bs, tr_nw, tr_pf, tr_pw, tr_dl = dl_tr
    te_bs, te_nw, te_pf, te_pw = dl_te
    u_bs, u_nw, u_pf, u_pw = dl_u
    uo_bs, uo_nw, uo_pf, uo_pw = dl_uo
    if args.dataset == 'STL':
        train_transform = utils.make_train_transform(normalize=args.image_class)
        test_transform = utils.make_test_transform(normalize=args.image_class)
        train_data = utils.STL10_Index(root=args.data, split='train+unlabeled', transform=train_transform)
        memory_data = utils.STL10(root=args.data, split='train', transform=train_transform, target_transform=target_transform)
        test_data = utils.STL10(root=args.data, split='test', transform=test_transform, target_transform=target_transform)
    elif args.dataset == 'CIFAR10':
        train_transform = utils.make_train_transform(normalize=args.image_class)
        test_transform = utils.make_test_transform(normalize=args.image_class)
        train_data = utils.CIFAR10_Index(root=args.data, train=True, transform=train_transform, target_transform=target_transform, download=True)
        memory_data = utils.CIFAR10(root=args.data, train=True, transform=train_transform, target_transform=target_transform)
        test_data = utils.CIFAR10(root=args.data, train=False, transform=test_transform, target_transform=target_transform)
    elif args.dataset == 'CIFAR100':
        train_transform = utils.make_train_transform(normalize=args.image_class)
        test_transform = utils.make_test_transform(normalize=args.image_class)
        train_data = utils.CIFAR100_Index(root=args.data, train=True, transform=train_transform, target_transform=target_transform)
        memory_data = utils.CIFAR100(root=args.data, train=True, transform=train_transform, target_transform=target_transform)
        test_data = utils.CIFAR100(root=args.data, train=False, transform=test_transform, target_transform=target_transform)
    elif args.dataset == 'ImageNet':
        train_transform = utils.make_train_transform(image_size, randgray=not args.norandgray, normalize=args.image_class)
        test_transform = utils.make_test_transform(normalize=args.image_class)
        train_data  = utils.Imagenet_idx(root=args.data + '/train', transform=train_transform, target_transform=target_transform, class_to_idx=class_to_idx)
        memory_data = utils.Imagenet_idx(root=args.data + '/train', transform=train_transform,  target_transform=target_transform, class_to_idx=class_to_idx)
        test_data   = utils.Imagenet(root=args.data     + '/test',  transform=test_transform,  target_transform=target_transform, class_to_idx=class_to_idx)
        val_data    = utils.Imagenet(root=args.data     + '/val',   transform=test_transform,  target_transform=target_transform, class_to_idx=class_to_idx)
        
    # pretrain model
    assert (args.pretrain_path is None) or (args.pretrain_path is not None and os.path.isfile(args.pretrain_path)), f"pretrain file {args.pretrain_path} is missing"
    if args.pretrain_path is not None and os.path.isfile(args.pretrain_path):
        msg = []
        print("=> loading pretrained checkpoint '{}'".format(args.pretrain_path), end="")
        checkpoint = torch.load(args.pretrain_path, map_location=device, weights_only=False)
        if 'state_dict' in checkpoint.keys():
            state_dict = checkpoint['state_dict']
            print(f" Epoch: {checkpoint['epoch']}")
        else:
            state_dict = checkpoint
            print(" Epoch: N/A")
    else:
        state_dict = None
        print('Using default model')

    c = len(train_data.classes) if args.dataset != "ImageNet" else args.class_num
    print('# Classes: {}'.format(c))

    # model setup and optimizer config
    ssl_type = args.ssl_type.lower()
    classifier_dim = c if args.loss_unsplit_type else None
    if ssl_type == 'moco' or ssl_type == 'mocosupcon':       
        arms_blueprints = {"projection": partial(create_mlp, output_dim=feature_dim, hidden_dims=[512], norm_layer=nn.BatchNorm1d, bias=[False, True],
                                                       last_layer_norm=False, last_layer_act=False)
        }
        shortcuts = {'g': 'projection'}
        
    elif ssl_type == 'simsiam':
        arms_blueprints = {"projector": partial(create_mlp, output_dim=feature_dim, hidden_dims=[512], norm_layer=nn.BatchNorm1d, bias=[False, False, False],
                                                            last_layer_norm=True, last_layer_act=False, 
                                                            norm_params=[{"affine": True}, {"affine": True}, {"affine": False}]),  
                           "predictor": partial(create_mlp, output_dim=feature_dim, hidden_dims=[feature_dim/2], norm_layer=nn.BatchNorm1d, bias=[False, True],
                                                            last_layer_norm=False, last_layer_act=False)
        }
        shortcuts = {'g': 'projector', 'h': 'predictor'}

    else:
        raise NotImplemented

    if classifier_dim:
        arms_blueprints.update({"classifier": partial(create_mlp, output_dim=classifier_dim, bias=True)})
        shortcuts.update({'fc': 'classifier'})

    # Mask
    # FIX ME!!!! Add mask sparsity loss
    mask_fun = Mask(args.mask_nonlinearity, tau=args.gumbel_tau, soft=args.gumbel_soft)
    mask_blueprint = partial(MaskModule, mask_fun, trainable=args.opt_mask)

    model = MultiArmModel(backbone_name='resnet50', mask_blueprint=mask_blueprint, arms_blueprints=arms_blueprints, in_transform=None, out_transforms=None, 
             shortcuts=shortcuts, image_class=image_class, state_dict=state_dict).cuda()

    if state_dict is not None:
        print("<= loaded pretrained checkpoint '{}'".format(args.pretrain_path))

    model = nn.DataParallel(model)

    if ssl_type == 'moco' or ssl_type == 'mocosupcon':
        model_momentum = copy.deepcopy(model)
        for p in model_momentum.parameters():
            p.requires_grad = False
        momentum = args.momentum              # momentum for model_momentum
        queue_size = args.queue_size
        queue = utils.FeatureQueue(queue_size, feature_dim, device=device, dtype=torch.float32, indices=True)
    elif ssl_type == 'simsiam':
        model_momentum = None
        queue = None
        momentum = None

    ema = utils.MovingAverage(0.95, oneminusema_correction=False, active=args.ema)
    
    initial_weights = {'penalty': torch.tensor(1.0, dtype=torch.float, device=device)}
    if args.penalty_cont > 0:
        initial_weights['loss'] = torch.tensor(1.0, dtype=torch.float, device=device)
    if args.unsplit_cont and (args.penalty_unsplit_cont > 0):
        initial_weights['loss_unsplit'] = torch.tensor(1.0, dtype=torch.float, device=device)
    gradnorm_balancer = gn.GradNormLossBalancer(initial_weights, alpha=args.gradnorm_alpha, device=device, smoothing=False, 
                            tau=args.gradnorm_tau, eps=1e-8, debug=args.gradnorm_debug, beta=args.gradnorm_beta, 
                            avgG_detach_frac=args.gradnorm_avgG_detach_frac, Gscaler=args.gradnorm_Gscaler, 
                            gradnorm_loss_type=args.gradnorm_loss_type, 
                            gradnorm_loss_lambda=args.gradnorm_loss_lambda, huber_delta=args.gradnorm_huber_delta)

    def get_optimizer_params(model, ssl_type):
        params = []
        if ssl_type == "simsiam":
            params.append({'params': model.module.f.parameters(), 'lr': args.featurizer_lr if args.featurizer_lr > 0 else args.lr})
            params.append({'params': model.module.arms['projector'].parameters(), 'lr': args.projector_lr if args.projector_lr > 0 else args.lr})
            params.append({'params': model.module.arms['predictor'].parameters(), 'lr': args.predictor_lr if args.predictor_lr > 0 else args.lr})
            params.append({'params': model.module.arms['classifier'].parameters(), 'lr': args.lr})
        else:
            params.append({'params': model.module.f.parameters(), 'lr': args.featurizer_lr if args.featurizer_lr > 0 else args.lr})
            params.append({'params': model.module.arms['projection'].parameters(), 'lr': args.projector_lr if args.projector_lr > 0 else args.lr})
            params.append({'params': model.module.arms['classifier'].parameters(), 'lr': args.lr})
        return params

    if args.opt == "Adam":
        params = get_optimizer_params(model, ssl_type)
        optimizer = optim.Adam(params, weight_decay=args.weight_decay, betas=args.betas)
        gradnorm_optimizer = optim.Adam(gradnorm_balancer.parameters(), lr=args.gradnorm_lr, weight_decay=args.gradnorm_weight_decay, betas=args.gradnorm_betas)        
    elif args.opt == 'SGD':
        optimizer          = optim.SGD(model.parameters(),             lr=args.lr, weight_decay=args.weight_decay, momentum=args.SGD_momentum)
        gradnorm_optimizer = optim.SGD(gradnorm_balancer.parameters(), lr=args.gradnorm_lr, weight_decay=args.weight_decay, momentum=args.SGD_momentum)

    # optionally resume from a checkpoint
    best_acc1 = 0
    best_epoch = 0
    resumed = False
    start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            (model, model_momentum, optimizer, queue,
             start_epoch, best_acc1, best_epoch,
             updated_split, updated_split_all, ema_, gradnorm_balancer, gradnorm_optimizer) = \
                load_checkpoint(args.resume, model, model_momentum, optimizer, gradnorm_balancer, gradnorm_optimizer, classifier_not_needed=classifier_dim is None)
 
            if not args.baseline:
                if (ema_ is not None) and (args.ema == 'retain'): # exists in checkpoint
                    ema = ema_
                ema.set_active(args.ema) # set to what the user has currently set
                # gradnorm restores only attributes needed to continue running. arguments are taken from  user args
                gradnorm_balancer.set_tau(args.gradnorm_tau) # always set tau to currently provided value; also converts None to values

            # use current LR, not the one from checkpoint
            params = get_optimizer_params(model, ssl_type)
            for pind, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = params[pind]['lr']

            for param_group in gradnorm_optimizer.param_groups:
                param_group['lr'] = args.gradnorm_lr 
            resumed = True
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
                    
    # training loop
    # start epoch is what the user provided, if provided, or from checkpoint, if exists, or 1 (default)
    start_epoch = args.start_epoch if args.start_epoch else start_epoch
    epoch = start_epoch # used from train_partition()
    print(f"start epoch {start_epoch}")

    if args.evaluate is not None:
        print(f"Starting evaluation name: {args.name}")
        if len(args.evaluate) == 0:
            args.evaluate = ['val', 'test']
        if 'val' in args.evaluate:
            print('eval on val data')
            val_loader = DataLoader(val_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=True, 
                pin_memory=True, persistent_workers=te_pw)
            val_acc_1, val_acc_5, val_macro_acc = test(model, val_loader, args, num_classes=c, progress=True, prefix="Val:")
        if 'test' in args.evaluate:
            print('eval on test data')
            test_loader = DataLoader(test_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=True, 
                pin_memory=True, persistent_workers=te_pw)
            test_acc_1, test_acc_5, test_macro_acc = test(model, test_loader, args, num_classes=c, progress=True, prefix="Test:")
        exit()

    if not args.resume and os.path.exists(log_file):
        os.remove(log_file)            
    
    if args.gradnorm_rescale_weights:
        gradnorm_balancer.rescale_weights()
    
    kwargs = {'ema': ema, 'gradnorm_balancer': gradnorm_balancer, 'gradnorm_optimizer': gradnorm_optimizer, 'log_file': log_file}
    if ssl_type == 'moco' or ssl_type == 'mocosupcon':
        kwargs.update({'net_momentum': model_momentum, 'queue': queue, 'temperature': temperature, 'momentum': momentum})
    elif ssl_type == 'simsiam':
        pass
        
    if args.loss_unsplit_type == 'CEweighted': # weight per-class loss w/ its inverse frequency
        labels = train_data.targets if isinstance(train_data.targets, torch.Tensor) else torch.tensor(train_data.targets)
        labels = target_transform(labels) if target_transform else labels
        counts = torch.bincount(labels)
        class_weights = counts.sum() / counts.float() / args.class_num  # use inverse frequency
        kwargs['CEweights'] = class_weights

    def shutdown_loader(loader):
        """Shutdown and release a DataLoader and its workers immediately."""
        if loader is None:
            return None
        it = getattr(loader, "_iterator", None)
        if it is not None:
            it._shutdown_workers()
        return None

    # Cluster creation / loading order:
    # 0. Load given cluster (see loading order above)
    # 1. Load most recent in cluster save directory, if exists
    # 2. Create cluster from loaded model.
    #    Suffix reflects the model used: 'resumed', 'pretrained', 'default'
    
    #### Process Cluster

    if args.cluster_path is None: # specific cluster not given
        directory = f'misc/{args.name}'
        pattern = 'env_ref_set_*' 

        # Find matching files
        files = glob.glob(os.path.join(directory, pattern))

        # Sort by modification time
        files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
        fp_exist = None
        if files_sorted:
            fp_exist = files_sorted[0]
        
        if args.resume and resumed:
            hash_object = hashlib.sha256(args.resume.encode())
            hex_dig = hash_object.hexdigest()
            suffix = hex_dig + '_resumed'
        elif args.pretrain_path is not None and os.path.isfile(args.pretrain_path):
            hash_object = hashlib.sha256(args.pretrain_path.encode())
            hex_dig = hash_object.hexdigest()
            suffix = hex_dig + '_pretrained'
        else:
            suffix = 'default'
        fp_new = os.path.join(directory, 'env_ref_set_' + suffix)
        
        if args.only_cluster or not fp_exist:
            fp = fp_new
        else:
            fp = fp_exist
            
    else:
        directory = f'misc/{args.name}'
        fp = args.cluster_path
        
    fp_dist = os.path.join(directory, 'env_ref_dist')
    
    if args.only_cluster or not os.path.exists(fp): # recalculate cluster OR cluster doesn't exist
        memory_loader = DataLoader(memory_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
            pin_memory=False, persistent_workers=te_pw)
        # Cannot use end="" b/c cal_cosine_distance prints progress bar and overwrites its
        if args.only_cluster:
            print('Recalculation of cluster file requested... ')
        else:
            print('No cluster file, creating... ')
        if args.cluster_save_dist: # save cluster distances
            env_ref_set, dist = utils_cluster.cal_cosine_distance(model, memory_loader, args.class_num, temperature=args.cluster_temp, 
                anchor_class=None, class_debias_logits=True, return_dist=True, K=args.num_clusters)
            os.makedirs(os.path.dirname(fp_dist), exist_ok=True)
            # dist is a dictionary with anchor classes as keys of similarity scores
            torch.save(dist, fp_dist)
            print(f"Cluster distances saved in {fp_dist}")
        else:
            env_ref_set = utils_cluster.cal_cosine_distance(model, memory_loader, args.class_num, temperature=args.cluster_temp, 
                anchor_class=None, class_debias_logits=True, K=args.num_clusters)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        torch.save(env_ref_set, fp)
        print(f'cluster {fp} ready!') 
        if args.only_cluster:
            exit(1)
        
        memory_loader = shutdown_loader(memory_loader)
        gc.collect()              # run Python's garbage collector
    else:
        env_ref_set = torch.load(fp)
        print(f'Cluster {fp} loaded.')
        assert len(env_ref_set[0]) == args.num_clusters, "Num clusters in cluster file {} != num_clusters {}".format(len(env_ref_set[0]), args.num_clusters)
        assert args.clusters_to_use is None or \
            max(args.clusters_to_use) <= args.num_clusters-1, "Largest cluster to use {} must be < {}".format(max(args.clusters_to_use), args.num_clusters)

    # 'partitions' should be a list of splits, each one a tensor w/ dim=1 equal to the number of environment and dim=0 equal to the number of samples
    # 'env_ref_set' is a dictionary w/ key equal to class index and value a tuple w/ size equal to the number of envs each one a tensor of samples' indices  
    def convert_env_ref_set_2_partitions(env_ref_set):
        partitions = []
        for cid, class_envs in env_ref_set.items():
            num_samples = [len(e) for e in class_envs]
            min_size = min(num_samples)
            partitions.append(torch.stack([e[:min_size] for e in class_envs], dim=-1))
        return partitions

    partitions = convert_env_ref_set_2_partitions(env_ref_set).to(device)

    train_loader = None

    for epoch in range(start_epoch, epochs + 1):
        if train_loader is None:
            train_loader = DataLoader(train_data, batch_size=tr_bs, num_workers=tr_nw, prefetch_factor=tr_pf, shuffle=True, 
                                pin_memory=True, persistent_workers=tr_pw, drop_last=tr_dl)

        train_loss = train_env(model, train_loader, optimizer, partitions, tr_bs, epoch, args, **kwargs)

        # eval model every test_freq/val_freq and last epochs
        if (                                 (epoch >= args.test_freq) and ((epoch % args.test_freq == 0) or (epoch == epochs))) or \
           ((args.dataset == 'ImageNet') and (epoch >= args.val_freq)  and ((epoch % args.val_freq == 0)  or (epoch == epochs))):
            if train_loader is not None:
                train_loader = shutdown_loader(train_loader)
                gc.collect()

        if (epoch >= args.test_freq) and ((epoch % args.test_freq == 0) or (epoch == epochs)): # eval model every test_freq epochs
            test_loader = DataLoader(test_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=True, 
                pin_memory=False, persistent_workers=te_pw)
            test_acc_1, test_acc_5, test_macro_acc = test(model, feauture_bank, test_loader, args, num_classes=c, progress=True, prefix="Test:")
            test_loader = shutdown_loader(test_loader)
            gc.collect()              # run Python's garbage collector
            txt_write = open("results-eqinv/{}/{}/{}".format(args.dataset, args.name, 'knn_result.txt'), 'a')
            txt_write.write('\ntest_acc@1: {}, test_acc@5: {}, test_macro_acc: {}'.format(test_acc_1, test_acc_5, test_macro_acc))
            torch.save(model.state_dict(), 'results-eqinv/{}/{}/model_{}.pth'.format(args.dataset, args.name, epoch))

        if (epoch >= args.val_freq) and ((epoch % args.val_freq == 0) or (epoch == epochs)) and (args.dataset == 'ImageNet'):
            # evaluate on validation set
            val_loader = DataLoader(val_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=True, 
                pin_memory=False, persistent_workers=te_pw)
            acc1, _, _ = test(model, feauture_bank, feature_labels, val_loader, args, num_classes=c, progress=True, prefix="Val:")
            val_loader = shutdown_loader(val_loader)
            gc.collect()              # run Python's garbage collector

            # remember best acc@1 & best epoch and save checkpoint
            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
                best_epoch = epoch
        else:
            is_best = False

        if (epoch % args.checkpoint_freq == 0) or (epoch == epochs):
            cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

            utils.atomic_save({
                'epoch':                epoch,
                'state_dict':           model.state_dict(),
                'best_acc1':            best_acc1,
                'best_epoch':           best_epoch,
                'optimizer':            optimizer.state_dict(),
                'state_dict_momentum':  model_momentum.state_dict() if model_momentum else None,
                'queue':                queue,
                'state_dict_gradnorm':  gradnorm_balancer.state_dict(),
                'gradnorm_optimizer':   gradnorm_optimizer.state_dict(),
                "rng_dict": {
                    "rng_state":        torch.get_rng_state(),
                    "cuda_rng_state":   cuda_rng_state,
                    "numpy_rng_state":  np.random.get_state(),
                    "python_rng_state": random.getstate(),
                },
                'ema':                  ema,
            }, is_best, filename='{}/{}/checkpoint.pth.tar'.format(args.save_root, args.name))
