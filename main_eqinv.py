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
from trainenv import train_env, CELossModule, MoCoLossModule, MoCoSupConLossModule, SimSiamLossModule, CE_IRMCalculator, SimSiamIRMCalculator, VRExCalculator  

from functools import partial

import utils_cluster

sys.modules['__main__'].FeatureQueue = utils.FeatureQueue

def build_losses_and_penalty_dict(args, net, class_weights=None, moco_dict=None):
    loss_type         = getattr(args, 'ssl_type', 'moco').lower()
    loss_type_unsplit = getattr(args, 'ssl_type_unsplit', 'moco').lower()
    penalty_type      = getattr(args, 'penalty_type', 'irm').lower()
    loss_CE_type      = getattr(args, 'loss_CE_type', None)
    loss_CE_type      = loss_CE_type.lower() if loss_CE_type is not None else None

    if loss_CE_type == 'ce' or loss_CE_type == 'ceweighted':
        LossCEModule = CELossModule
    elif loss_CE_type is None:
        LossCEModule = None
    else:
        raise ValueError(f"Unknown loss_CE_type: {loss_CE_type}")

    if loss_type == 'moco':
        LossModule = MoCoLossModule
    elif loss_type == 'mocosupcon':
        LossModule = MoCoSupConLossModule
    elif loss_type == 'simsiam':
        LossModule = SimSiamLossModule
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if loss_type_unsplit == 'moco':
        LossUnsplitModule = MoCoLossModule
    elif loss_type_unsplit == 'mocosupcon':
        LossUnsplitModule = MoCoSupConLossModule
    elif loss_type_unsplit == 'simsiam':
        LossUnsplitModule = SimSiamLossModule
    else:
        raise ValueError(f"Unknown loss_type: {loss_type_unsplit}")

    kwargs = moco_dict
    kwargs.update({'CEweights': class_weights})
    if (not args.domained_clusters) and (loss_type == 'mocosupcon'):
        def filter_indices(idxs, labels, partition, **kwargs):
            idxs = idxs[labels==partition]
            return idxs
        kwargs.update({'filter_indices': filter_indices})

    loss_and_penalties_dict = {}

    if LossCEModule is not None:
        loss_CE_module = LossCEModule(net, debug=args.debug, detached_backbone=False, **kwargs) 
    else:
        loss_CE_module =  None
    loss_and_penalties_dict['loss_CE_module'] = loss_CE_module

    loss_module = LossModule(net, debug=args.debug, detached_backbone=True, projector=False, queue=kwargs['queue_noproj'], **kwargs) 
    loss_and_penalties_dict['loss_module'] = loss_module

    loss_unsplit_module = LossUnsplitModule(net, debug=args.debug, detached_backbone=False, projector=True, queue=kwargs['queue_proj'], **kwargs) 
    loss_and_penalties_dict['loss_unsplit_module'] = loss_unsplit_module

    # IRM calculator selection
    if penalty_type == 'irm':
        if loss_type   == 'moco' or loss_type == 'mocosupcon':
            PenaltyCalculator = CE_IRMCalculator
        elif loss_type == 'simsiam':
            PenaltyCalculator = SimSiamIRMCalculator
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
    elif penalty_type == 'vrex':
        PenaltyCalculator = VRExCalculator
    else:
        raise ValueError(f"Unknown penalty_type: {penalty_type}")

    penalty_calculator = PenaltyCalculator(loss_module, irm_temp=args.irm_temp, debug=args.debug, **kwargs)
    loss_and_penalties_dict['penalty_calculator'] = penalty_calculator
    
    return loss_and_penalties_dict

def setup_moco_model(model, args):
    ssl_type = args.ssl_type.lower()
    device = next(model.parameters()).device
    if ssl_type == 'moco' or ssl_type == 'mocosupcon':
        model_momentum = copy.deepcopy(model)
        for p in model_momentum.parameters():
            p.requires_grad = False
        momentum = args.momentum              # momentum for model_momentum
    elif ssl_type == 'simsiam':
        model_momentum = None
    else:
        raise ValueError(f"Unknown ssl_type: {ssl_type}")
        
    return model_momentum

def build_arms_blueprints(args, num_classes):
    ssl_type = args.ssl_type.lower()
    classifier_dim = num_classes
    if ssl_type == 'moco' or ssl_type == 'mocosupcon':       
        arms_blueprints = {"projector": partial(create_mlp, output_dim=args.feature_dim, hidden_dims=[512], norm_layer=nn.BatchNorm1d, bias=[False, True],
                                                       last_layer_norm=False, last_layer_act=False)
        }
        shortcuts = {'g': 'projector'}
        
    elif ssl_type == 'simsiam':
        arms_blueprints = {"projector": partial(create_mlp, output_dim=args.feature_dim, hidden_dims=[512, 512], norm_layer=nn.BatchNorm1d, bias=[False, False, False],
                                                            last_layer_norm=True, last_layer_act=False, 
                                                            norm_kwargs=[{"affine": True}, {"affine": True}, {"affine": False}]),  
                           "predictor_proj": partial(create_mlp, input_dim=args.feature_dim, output_dim=args.feature_dim, hidden_dims=[int(args.feature_dim/2)], 
                                                            norm_layer=nn.BatchNorm1d, bias=[False, True],
                                                            last_layer_norm=False, last_layer_act=False)
        }
        shortcuts = {'g': 'projector', 'h_proj': 'predictor_proj'}

    else:
        raise ValueError(f"Unknown ssl_type: {ssl_type}")

    arms_blueprints.update({"classifier": partial(create_mlp, output_dim=classifier_dim, bias=True)})
    shortcuts.update({'fc': 'classifier'})
    
    return arms_blueprints, shortcuts

def setup_gradnorm_balancer(args, device):
    initial_weights = {'penalty': torch.tensor(1.0, dtype=torch.float, device=device)}
    if args.penalty_cont > 0:
        initial_weights['loss'] = torch.tensor(1.0, dtype=torch.float, device=device)
    if args.unsplit_cont and (args.penalty_unsplit_cont > 0):
        initial_weights['loss_unsplit'] = torch.tensor(1.0, dtype=torch.float, device=device)
    if args.penalty_CE > 0:
        initial_weights['loss_CE'] = torch.tensor(1.0, dtype=torch.float, device=device)
    gradnorm_balancer = gn.GradNormLossBalancer(initial_weights, alpha=args.gradnorm_alpha, device=device, smoothing=False, 
                            tau=args.gradnorm_tau, eps=1e-8, debug=args.gradnorm_debug, beta=args.gradnorm_beta, 
                            avgG_detach_frac=args.gradnorm_avgG_detach_frac, Gscaler=args.gradnorm_Gscaler, 
                            gradnorm_loss_type=args.gradnorm_loss_type, 
                            gradnorm_loss_lambda=args.gradnorm_loss_lambda, huber_delta=args.gradnorm_huber_delta)
    return gradnorm_balancer

def get_feature_bank(net, memory_data_loader, args, progress=False, prefix="Test:", mask_u=None, masked_features=True):
    net.eval()
    
    if isinstance(memory_data_loader.dataset, Subset):
        dataset = memory_data_loader.dataset.dataset
        idcs    = memory_data_loader.dataset.indices
    else:
        dataset = memory_data_loader.dataset
        idcs    = list(range(len(dataset)))
    transform = dataset.transform
    feature_bank = []
    
    with torch.no_grad():
        # generate feature bank
        bar_format = '{l_bar}{bar:' + str(args.bar) + '}{r_bar}' #{bar:-' + str(args.bar) + 'b}'
        if progress:
            feature_bar = tqdm(memory_data_loader,
                total=len(memory_data_loader),
                ncols=args.ncols,               # total width available
                dynamic_ncols=False,            # disable autosizing
                bar_format=bar_format,          # request bar width
                desc='get_feature_bank(), memory: Feature extracting'
            )
        else:
            feature_bar = memory_data_loader
        for data, _, _ in feature_bar:
            data = data.cuda(non_blocking=True)

            if transform is not None:
                data = transform(data)
                
            feature = net.module.backbone(data)
            feature = utils.safe_normalize(feature, dim=-1)
            if masked_features:
                mask_activation = net.module.mask_fun.activation(u=mask_u)
                feature = feature * mask_activation
                #feature = utils.safe_normalize(feature, dim=-1)
            feature_bank.append(feature)
        #end for data, _, _ in feature_bar:

        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous() # places feature_bank on cuda
        # [N]
        if hasattr(dataset, "labels"):
            labels = [dataset.labels[i] for i in indcs]
        else:
            targets = [dataset.targets[i] for i in idcs]
            if dataset.target_transform is not None:
                labels = [dataset.target_transform(t) for t in targets]
            else:
                labels = targets       
        feature_labels = torch.tensor(labels, device=feature_bank.device)

    return feature_bank, feature_labels

def test_knn(net, feature_bank, feature_labels, test_data_loader, num_classes, args, progress=False, prefix="Test:", mask_u=None, masked_features=True):
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
    
        if isinstance(test_data_loader.dataset, Subset):
            dataset = test_data_loader.dataset.dataset
            idcs    = test_data_loader.dataset.indices
        else:
            dataset = test_data_loader.dataset
            idcs    = list(range(len(dataset)))
        transform = dataset.transform
        target_transform = dataset.target_transform
    
        # For macro-accuracy computation
        per_class_correct = torch.zeros(num_classes, dtype=torch.long, device=feature_bank[0].device)
        per_class_total   = torch.zeros(num_classes, dtype=torch.long, device=feature_bank[0].device)

        if mask_u is None:
            mask_u = net.module.mask_fun.sample().detach()

        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            
            if transform is not None:
                data = transform(data)

            feature = net.module.backbone(data)
            feature = utils.safe_normalize(feature, dim=-1)
            if masked_features:
                mask_activation = net.module.mask_fun.activation(u=mask_u)
                features = feature * mask_activation
                #features = utils.safe_normalize(feature, dim=-1)
            
            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            # feature & feature_bank are normalized
            sim_matrix = torch.mm(feature, feature_bank) # places sim_matrix on cuda
            # [B, K]
            # A namedtuple of (values, indices) is returned with the values and indices 
            # of the largest k elements of each row of the input tensor in the given dimension dim.
            sim_weight, sim_indices = sim_matrix.topk(k=args.k, dim=-1)
            # [B, K]
            # The "original size" refers to the size of that dimension in the input tensor, after any necessary 
            # prepending of 1s on the left to match number of requested dimensions.    
            # For each sample, picks the top-k labels that correspond to the similarity matrix of that sample and the feature bank 
            #                                                 (B,N)
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / args.knn_temp).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * args.k, num_classes, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, num_classes) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            # predicted class is the top-1 label
            pred = pred_labels[:, 0]   # [B]

            # Loop-free update of per-class counts
            # For each class c: count how many predictions & targets match
            for cls in range(num_classes):
                mask = (target == cls)
                if mask.any():
                    per_class_total[cls] += mask.sum()
                    per_class_correct[cls] += (pred[mask] == cls).sum()

            if progress:
                # Avoid division by zero in rare cases
                valid = per_class_total > 0
                macro_acc = (per_class_correct[valid].float() / per_class_total[valid].float()).mean().item()
                test_bar.set_description('KNN {} Epoch [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}% Macro-Acc:{:.2f}%'
                                         .format(prefix, epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100, macro_acc * 100))

        # end for data, _, target in test_bar

        # Avoid division by zero in rare cases
        valid = per_class_total > 0
        macro_acc = (per_class_correct[valid].float() / per_class_total[valid].float()).mean().item()

    return total_top1 / total_num * 100, total_top5 / total_num * 100, macro_acc * 100

# test for one epoch
def test(net, test_data_loader, args, num_classes, progress=False, prefix="Test:", mask_u=None):
    net.eval()
       
    total_top1, total_top5, total_num = 0.0, 0.0, 0
    with torch.no_grad():
        # loop test data to predict the label
        bar_format = '{l_bar}{bar:' + str(args.bar) + '}{r_bar}' #{bar:-' + str(args.bar) + 'b}'
        
        if progress:
            test_bar = tqdm(test_data_loader,
                total=len(test_data_loader),
                ncols=args.ncols,               # total width available
                dynamic_ncols=False,            # disable autosizing
                bar_format=bar_format,          # request bar width
                file=sys.stdout,    # Ensures it uses standard output
                mininterval=1.0,   # Only updates the UI every 10 seconds
                maxinterval=2.0,   # Limits the maximum refresh rate
                ascii=True,         # Uses simple chars (less likely to break the socket)
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
        masked_feature_list = []
        pred_labels_list = []
        pred_scores_list = []
        target_list = []
        target_raw_list = []
        device = next(net.parameters()).device

        # For macro-accuracy computation
        per_class_correct = torch.zeros(num_classes, dtype=torch.long, device=device)
        per_class_total   = torch.zeros(num_classes, dtype=torch.long, device=device)

        if mask_u is None:
            mask_u = net.module.mask_fun.sample().detach()

        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            
            if transform is not None:
                data = transform(data)

            target_raw = target
            if args.extract_features and target_transform is not None:
                target = target_transform(target_raw).cuda(non_blocking=True)

            features = net.module.backbone(data)
            features = utils.safe_normalize(features, dim=-1)
            mask_activation = net.module.mask_fun.activation(u=mask_u)
            masked_features = features * mask_activation
            # Gemini says to normalize
            masked_features = utils.safe_normalize(masked_features, dim=-1)
            
            out = net.module.fc(masked_features)

            total_num += data.size(0)

            pred_labels = out.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            # predicted class is the top-1 label
            pred = pred_labels[:, 0]   # [B]

            # Loop-free update of per-class counts
            # For each class c: count how many predictions & targets match
            for cls in range(num_classes):
                mask = (target == cls)
                if mask.any():
                    per_class_total[cls] = per_class_total[cls] + mask.sum()
                    per_class_correct[cls] = per_class_correct[cls] + (pred[mask] == cls).sum()

            if progress:
                # Avoid division by zero in rare cases
                valid = per_class_total > 0
                macro_acc = (per_class_correct[valid].float() / per_class_total[valid].float()).mean().item()
                test_bar.set_description('{} Epoch [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}% Macro-Acc:{:.2f}%'
                                         .format(prefix, epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100, macro_acc * 100))

            # compute output
            if args.extract_features:
                feature_list.append(features)
                masked_feature_list.append(masked_features)
                target_list.append(target)
                target_raw_list.append(target_raw)
                pred_labels_list.append(pred_labels)
                pred_scores_list.append(out)

        # end for data, _, target in test_bar
        
        # Avoid division by zero in rare cases
        valid = per_class_total > 0
        macro_acc = (per_class_correct[valid].float() / per_class_total[valid].float()).mean().item()

        if feature_list:
            features = torch.cat(feature_list, dim=0)
            masked_features = torch.cat(masked_feature_list, dim=0)
            target = torch.cat(target_list, dim=0)
            target_raw = torch.cat(target_raw_list, dim=0)
            pred_labels = torch.cat(pred_labels_list, dim=0)
            pred_scores = torch.cat(pred_scores_list, dim=0)

            # Save to file
            if "Test" in prefix:
                prefix = "test"
            elif "Val" in prefix:
                prefix = "val"
            elif "Train" in prefix:
                prefix = "train"
            directory = f'results-eqinv/{args.dataset}/{args.name}'
            fp = os.path.join(directory, f"{prefix}_features_dump.pt")       
            os.makedirs(os.path.dirname(fp), exist_ok=True)

            state = {
                'features':        features,
                'masked_features': masked_features,
                'mask':            mask_activation,
                'labels':          target,
                'labels_raw':      target_raw,
                'pred_labels':     pred_labels,
                'pred_scores':     pred_scores,
                'model_epoch':     epoch,
                'head_weights':    net.module.arms["classifier"][0].weight,  # shape: (num_classes, embed_dim)
                'head_bias':       net.module.arms["classifier"][0].bias,    # shape: (num_classes,)
                'n_classes':       args.class_num,
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
    args = checkpoint.get("args", None)


    # Restore main model
    msg_model = model.load_state_dict(checkpoint["state_dict"], strict=False)
    msg_model = msg_model._asdict()
    # Don't care about if classifier not present, if not needed
    if classifier_not_needed:
        msg_model['unexpected_keys'] = [k for k in msg_model['unexpected_keys'] if not k.startswith('module.arms.classifier')]

    # Restore momentum model if applicable
    queue_proj = None
    queue_noproj = None
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

        if "queue_proj" in checkpoint and checkpoint["queue_proj"] is not None:
            queue_proj = checkpoint["queue_proj"]
        else:
            queue_proj = None
        if "queue_noproj" in checkpoint and checkpoint["queue_noproj"] is not None:
            queue_noproj = checkpoint["queue_noproj"]
        else:
            queue_noproj = None
    else:
        msg_momentum = "momentum encoder not used"
        queue_proj = None
        queue_noproj = None
        
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

    def restore_optimizer(optimizer, opt_state_dict, device, opt_string):
        print(f"Restoring optimizer {opt_string}")
        # 1. Capture the current params (the ones you want to keep)
        # This ensures your new mask is known to the system
        current_param_groups = optimizer.param_groups 

        # 2. Load the old state dict strictly
        # If the number of groups changed, this might throw an error. 
        # If it does, load ONLY the 'state' part.
        try:
            optimizer.load_state_dict(opt_state_dict)
        except ValueError:
            print(f"Optimizer {opt_string} group mismatch. Loading state values manually...")
            # Fallback: Load state buffers but keep current group structure
            optimizer.state.update(opt_state_dict["state"])

        # 3. Ensure every parameter in your current groups has a state entry
        # This is your "Initialization" logic, but safer:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p not in optimizer.state:
                    optimizer.state[p] = {
                        'step': torch.tensor(0.0).to(device), 
                        'exp_avg': torch.zeros_like(p),
                        'exp_avg_sq': torch.zeros_like(p)
                    }
                else:
                    # Move existing state to correct device
                    for k, v in optimizer.state[p].items():
                        if torch.is_tensor(v):
                            optimizer.state[p][k] = v.to(device)

    if "optimizer" in checkpoint and checkpoint["optimizer"] is not None and optimizer is not None:
        restore_optimizer(optimizer, checkpoint["optimizer"], device, "main")
                        
    if ("gradnorm_optimizer" in checkpoint) and (checkpoint["gradnorm_optimizer"] is not None) and (gradnorm_optimizer is not None):
        restore_optimizer(gradnorm_optimizer, checkpoint["gradnorm_optimizer"], device, "gradnorm")

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
    if queue_proj is not None:
        print("\tqueue_proj restored")
    if queue_noproj is not None:
        print("\tqueue_noproj restored")
    if gradnorm_balancer is not None:
        print("\tgradnorm load: {}".format(msg_gradnorm))

    print("<= loaded checkpoint '{}' (epoch {})".format(path, checkpoint.get("epoch", -1)))
    if args is not None:
        print("saved with args:")
        print(args)

    return model, model_momentum, optimizer, queue_proj, queue_noproj, start_epoch, best_acc1, best_epoch, updated_split, updated_split_all, ema, gradnorm_balancer, gradnorm_optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EqInv')
    parser.add_argument('--ssl_type', default='MoCo', type=str, choices=['MoCo', 'SimSiam', 'MoCoSupCon'], help='SSL type')    
    parser.add_argument('--ssl_type_unsplit', default='MoCo', type=str, choices=['MoCo', 'SimSiam', 'MoCoSupCon'], help='SSL type')    
    parser.add_argument('--ssl_type_partition', default='MoCoSupCon', type=str, choices=['MoCo', 'SimSiam', 'MoCoSupCon', 'SimCLR'], help='SSL type for partition')    
    parser.add_argument('--penalty_type', default='IRM', type=str, choices=['IRM', 'VREx'], help='Penalty type')        
    parser.add_argument('--penalty_sigma', default=None, type=float, help='Noise level to inject into penalty')        
    parser.add_argument('--grad_rotate', default=None, type=float, nargs=2, help='rotate gradients')      
    parser.add_argument('--loss_CE_type', required=True, type=str, choices=['CE', 'CEweighted'], help='Loss unsplit type')    
    parser.add_argument('--irm_temp', default=0.5, type=float, help='irm loss temperature')

    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature in SSL softmax')
    parser.add_argument('--moco_temperature', default=[0.5,0.5], type=float, nargs=2, help='Temperature used in softmax', metavar="[+temp, -temp]")
    parser.add_argument('--norandgray', action="store_true", default=False, help='skip rand gray transform')
    parser.add_argument('--random_aug', action="store_true", default=False, help='random_aug')

    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--knn_temp', default=0.5, type=float, help='Temperature used in KNN softmax')

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
    parser.add_argument('--CE_loss', action="store_true", default=False, help='do CE loss')
    parser.add_argument('--penalty_CE', default=1.0, type=float, help='CE loss weight')

    parser.add_argument('--retain_group', action="store_true", default=False, help='dummy')
    parser.add_argument('--Lscaler', default=1.0, type=float, help='Global scaler for losses gards')
    parser.add_argument('--penalty_iters', default=0, type=int, help='penalty weight start iteration')
    parser.add_argument('--increasing_weight', nargs=5, type=float, default=None, help='increasing penalty weight', 
            metavar='penalty_warmup, scale, speed, eps, debug')
    parser.add_argument('--env_num', default=2, type=int, help='number of environments in partition')
    parser.add_argument('--weight_env_eps', default=0., type=float, help='eps for per-env grad noise')

    parser.add_argument('--debug', action="store_true", default=False, help='debug?')
    parser.add_argument('--debug_dont_update_loss', action="store_true", default=False, help='debug?')
    parser.add_argument('--debug_print_loss', action="store_true", default=False, help='debug?')
    parser.add_argument('--debug_print_grads', action="store_true", default=False, help='debug?')
    parser.add_argument('--print_batch', action="store_true", default=False, help='print every batch')

    # image
    parser.add_argument('--image_size', type=int, default=224, help='image size')

    # domain/color in label
    parser.add_argument('--target_transform', type=str, default=None, help='a function definition to apply to target')
    parser.add_argument('--class_to_idx', type=str, default=None, help='a function definition to apply to class to obtain its index')
    parser.add_argument('--image_class', choices=['ImageNet', 'STL', 'CIFAR'], default='ImageNet', help='Image class, default=ImageNet')
    parser.add_argument('--class_num', default=1000, type=int, help='num of classes')
    parser.add_argument('--domains_path', type=str, default=None, help='domains file of training samples')
    parser.add_argument('--crossdomain_alpha', type=float, default=1.0, help='multiplier of cross domain positives')

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

    parser.add_argument('--evaluate', type=str, default=None, nargs="*", choices=['train', 'val', 'test', 'knn', 'masked'], help='only evaluate')
    parser.add_argument('--extract_features', action="store_true", help="extract features for post processing during evaluate")
    parser.add_argument('--split_train_for_test', type=float, default=None, nargs=2, help="fractions to split training data into train/val for evaluation")

    parser.add_argument('--opt', choices=['Adam', 'SGD'], default='Adam', help='Optimizer to use')
    parser.add_argument('--lr', default=0.001, type=float, help='LR')
    parser.add_argument('--SGD_momentum', default=0.9, type=float, help='LR')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay')
    parser.add_argument('--betas', default=[0.9, 0.999], type=float, nargs=2, help='Adam betas')

    parser.add_argument('--ema', type=str, default=None, choices=['reinit', 'retain'], help="adjust gradients w/ EMA")
    parser.add_argument('--gradnorm', action="store_true", help="use gradnorm")
    parser.add_argument('--gradnorm_epoch', default=0, type=int, help='gradnorm start epoch')
    parser.add_argument('--gradnorm_alpha', default=1.0, type=float, help='gradnorm alpha')
    parser.add_argument('--gradnorm_tau', default=None, nargs=2*4, type=str,
                        action=utils.ParseMixed, types=[str, float, str, float, str, float, str, float],
                        metavar='tau dictionary k-v pairs',    
                        help='loss divisors')
    parser.add_argument('--gradnorm_scalers', default=['loss_unsplit', 1.0, 'loss', 1.0, 'penalty', 1.0], nargs=2*4, type=str,
                        action=utils.ParseMixed, types=[str, float, str, float, str, float, str, float],
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
    parser.add_argument('--featurizer_lr', type=float, default=-1., help="featurizer LR")
    parser.add_argument('--mask_lr', type=float, default=-1., help="mask LR")
    parser.add_argument('--projector_lr', type=float, default=-1., help="projector LR")
    parser.add_argument('--predictor_lr', type=float, default=-1., help="predictor LR")
    parser.add_argument('--classifier_lr', type=float, default=-1., help="classifer LR")
    parser.add_argument('--bn_momentum', type=float, default=0.1, help="BN momentum")

    #### add mask
    parser.add_argument('--mask_nonlinearity', type=str, default='sigmoid', choices=['sigmoid', 'ident', 'gumbel'], help='type of non-linearity in mask')
    parser.add_argument('--opt_mask', action="store_true", default=False, help='optimize the mask')
    parser.add_argument('--mask_tau', type=float, default=1.0, help='tau for mask sigmoid')
    parser.add_argument('--gumbel_soft', action="store_true", help='soft gumbel')
    parser.add_argument('--mask_sparsity', type=int, default=None, help='sparsity K s.t. # of hot masks <= K')
    parser.add_argument('--mask_sparsity_weight', type=float, default=0.0, help='weight of sparsity loss')
    parser.add_argument('--mask_hard_sparsity_limit', action="store_true", help='if true, # masks always <= K')    
    parser.add_argument('--mask_save_freq', type=int, default=None, help='save mask frequency')
    parser.add_argument('--mask_sparsity_loss', type=str, default="L1/2", choices=['L1/2', 'Hoyer'], help='Loss to use for sparsity')

    # clustering
    parser.add_argument('--cluster_path', type=str, default=None, 
        help='path to cluster file. None means automatic creation ./misc/<name>/env_ref_set_<resumed|pretrained|default>')
    parser.add_argument('--only_cluster', action="store_true", help='only do clustering')
    parser.add_argument('--cluster_temp', type=float, default=0.1, help='temperature for clusteing') 
    parser.add_argument('--cluster_save_dist', action="store_true", help='save cluster distances in ./misc/<name>/env_ref_dist')
    parser.add_argument('--num_clusters', type=int, default=2, help='number of custer K') 
    parser.add_argument('--clusters_to_use', type=int, nargs='+', default=None, help='clusters to use out of K clusters') 
    parser.add_argument('--domained_clusters', action="store_true", help='clusters represent domains')
    

    parser.add_argument('--backbone_propagate', action="store_true", default=False, help='whether to propagate inv loss to backbone')
    parser.add_argument('--decimate_partitions', type=int, default=None, help='whether to decimate partitions')

    parser.add_argument('--train_transform', default='test', type=str, choices=['train', 'test', 'train_mixed']) # in LP train transfrom = test transfrom
    parser.add_argument('--test_transform', default='test', type=str, choices=['train', 'test'])
    parser.add_argument('--val_transform', default='test', type=str, choices=['train', 'test'])

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
    
    feature_dim, temperature, moco_temperature = args.feature_dim, args.temperature, args.moco_temperature
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
        train_transform = utils.make_train_transform(image_size, randgray=not args.norandgray, normalize=args.image_class, 
                            mixed=args.train_transform=='train_mixed', hard=args.random_aug)
        test_transform = utils.make_test_transform(normalize=args.image_class)
        
        transform   = train_transform if 'train' in args.train_transform else test_transform
        train_data  = utils.Imagenet_idx(root=args.data + '/train', transform=transform, target_transform=target_transform, class_to_idx=class_to_idx)
        memory_data = utils.Imagenet_idx(root=args.data + '/train', transform=transform,  target_transform=target_transform, class_to_idx=class_to_idx)
        transform   = train_transform if args.test_transform == 'train' else test_transform
        test_data   = utils.Imagenet(root=args.data     + '/test',  transform=transform,  target_transform=target_transform, class_to_idx=class_to_idx)
        transform   = train_transform if args.val_transform == 'train' else test_transform
        val_data    = utils.Imagenet(root=args.data     + '/val',   transform=transform,  target_transform=target_transform, class_to_idx=class_to_idx)
        
    # pretrain model
    assert (args.pretrain_path is None) or (args.pretrain_path is not None and os.path.isfile(args.pretrain_path)), f"pretrain file {args.pretrain_path} is missing"
    if args.pretrain_path is not None and os.path.isfile(args.pretrain_path):
        msg = []
        print("=> loading pretrained checkpoint '{}'".format(args.pretrain_path), end="")
        checkpoint = torch.load(args.pretrain_path, map_location=device, weights_only=False)
        if 'state_dict' in checkpoint.keys():
            state_dict = checkpoint['state_dict']
            print(f" Epoch {checkpoint['epoch']}")
        else:
            state_dict = checkpoint
            print(" Epoch N/A")
    else:
        state_dict = None
        print('Using default model')

    c = len(train_data.classes) if args.dataset != "ImageNet" else args.class_num
    print('# Classes: {}'.format(c))

    # model setup and optimizer config
    arms_blueprints, shortcuts = build_arms_blueprints(args, num_classes=c)

    # Mask
    mask_fun = Mask(args.mask_nonlinearity, tau=args.mask_tau, soft=args.gumbel_soft, K=args.mask_sparsity, hard_K=args.mask_hard_sparsity_limit)
    mask_blueprint = partial(MaskModule, mask_fun, trainable=args.opt_mask, device=device)

    # Model
    model = MultiArmModel(backbone_name='resnet50', mask_blueprint=mask_blueprint, arms_blueprints=arms_blueprints, in_transform=None, out_transforms=None, 
             shortcuts=shortcuts, image_class=image_class, state_dict=state_dict).cuda()
    if args.ssl_type.lower() == 'simsiam':
        arms_blueprints = {"predictor_noproj": partial(create_mlp, input_dim=model.feature_dim, output_dim=model.feature_dim, hidden_dims=[int(model.feature_dim/2)], 
                                                            norm_layer=nn.BatchNorm1d, bias=[False, True],
                                                            last_layer_norm=False, last_layer_act=False)
        }
        shortcuts = {'h_noproj': 'predictor_noproj'}
        model.add_arms(arms_blueprints=arms_blueprints, out_transforms=None, shortcuts=shortcuts)
    if state_dict is not None:
        print("<= loaded pretrained checkpoint '{}'".format(args.pretrain_path))

    model = nn.DataParallel(model)

    # MoCo momentum encoder
    model_momentum = setup_moco_model(model, args)
    queue_size = args.queue_size
    queue_proj = utils.FeatureQueue(queue_size, args.feature_dim,     device=device, dtype=torch.float32, indices=True)
    queue_noproj = utils.FeatureQueue(queue_size, model.module.feature_dim, device=device, dtype=torch.float32, indices=True)

    # EMA
    ema = utils.MovingAverage(0.95, oneminusema_correction=False, active=args.ema)
    
    # GradNorm
    gradnorm_balancer = setup_gradnorm_balancer(args, device)

    # Optimizers
    def get_optimizer_params(model, args):
        ssl_type = args.ssl_type.lower()
        params = []
        LRs = {}

        def set_lr(self_lr, default_lr, group, parameters):
            lr = self_lr if self_lr >= 0 else default_lr
            LRs[group] = lr
            params.append({'params': parameters, 'lr': lr, 'name': group})
                   
        set_lr(args.featurizer_lr, args.lr, 'backbone',   model.module.f.parameters())
        set_lr(args.classifier_lr, args.lr, 'classifier', model.module.arms['classifier'].parameters(),)
        set_lr(args.projector_lr,  args.lr, 'projector',  model.module.arms['projector'].parameters())
        if args.opt_mask:
            set_lr(args.mask_lr, args.lr, 'mask', model.module.mask_fun.parameters())
        if ssl_type == "simsiam":
            set_lr(args.predictor_lr, args.lr, 'predictor', model.module.arms['predictor'].parameters())
        return params, LRs

    if args.opt == "Adam":
        params, LRs = get_optimizer_params(model, args)
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
            (model, model_momentum, optimizer, queue_proj_, queue_noproj_,
             start_epoch, best_acc1, best_epoch,
             updated_split, updated_split_all, ema_, gradnorm_balancer, gradnorm_optimizer) = \
                load_checkpoint(args.resume, model, model_momentum, optimizer, gradnorm_balancer, gradnorm_optimizer, classifier_not_needed=False)
 
            # set LRs to current values
            for group in optimizer.param_groups:
                group_name = group.get('name')
                if group_name in LRs:
                    group['lr'] = LRs[group_name]                    
                else:
                    print(f"Unknown group {group_name} in LRs {LRs.keys()}")

            for gi, group in enumerate(gradnorm_optimizer.param_groups):
                group['lr'] = args.gradnorm_lr

            queue_proj = queue_proj_ or queue_proj
            queue_noproj     = queue_noproj_     or queue_noproj
            if not args.baseline:
                if (ema_ is not None) and (args.ema == 'retain'): # exists in checkpoint
                    ema = ema_
                ema.set_active(args.ema) # set to what the user has currently set
                # gradnorm restores only attributes needed to continue running. arguments are taken from  user args
                gradnorm_balancer.set_tau(args.gradnorm_tau) # always set tau to currently provided value; also converts None to values

            # set LRs to current values
            for group in optimizer.param_groups:
                group_name = group.get('name')
                if group_name in LRs:
                    group['lr'] = LRs[group_name]                    
                else:
                    print(f"Unknown group {group_name} in LRs {LRs.keys()}")

            for param_group in gradnorm_optimizer.param_groups:
                param_group['lr'] = args.gradnorm_lr 
            resumed = True
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    moco_dict = {'net_momentum': model_momentum, 'queue_proj': queue_proj, 'queue_noproj': queue_noproj, 'momentum': args.momentum,
                 'temperature': moco_temperature,}
    if args.domains_path is not None:
        domains = torch.load(args.domains_path, weights_only=False)
        domains = domains['partitions'][0]
        domains = torch.argmax(domains, dim=1).to(device)
        moco_dict.update({'domains': domains, 'crossdomain_alpha': args.crossdomain_alpha})

    losses_and_penalty_dict = build_losses_and_penalty_dict(args, model, class_weights=None, moco_dict=moco_dict)

    # training loop
    # start epoch is what the user provided, if provided, or from checkpoint, if exists, or 1 (default)
    start_epoch = args.start_epoch if args.start_epoch else start_epoch
    epoch = start_epoch # used from train_partition()
    print(f"start epoch {start_epoch}")

    if args.evaluate is not None and 'knn' in args.evaluate:
        print(f"Starting KNN evaluation name: {args.name}")
        
        mask_activation_noise = model.module.mask_fun.sample().detach()

        if args.split_train_for_test:
            mem_data = random_split(memory_data, args.split_train_for_test)
            memory_data = mem_data[0]
        memory_loader = DataLoader(memory_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
            pin_memory=True, persistent_workers=te_pw)
        feauture_bank, feature_labels = get_feature_bank(model, memory_loader, args, progress=True, prefix="Evaluate:", mask_u=mask_activation_noise,
                                            masked_features='masked' in args.evaluate)
        
        if args.split_train_for_test:
            print('eval on train data')
            train_loader = DataLoader(mem_data[1], batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
                pin_memory=True, persistent_workers=te_pw)
            train_acc_1, train_acc_5, train_macro_acc = test_knn(model, feauture_bank, feature_labels, train_loader, c, args, progress=True, prefix="Train:",
                    mask_u=mask_activation_noise, masked_features='masked' in args.evaluate)
        if 'val' in args.evaluate:
            print('eval on val data')
            val_loader = DataLoader(val_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=True, 
                pin_memory=True, persistent_workers=te_pw)
            val_acc_1, val_acc_5, val_macro_acc = test_knn(model, feauture_bank, feature_labels, val_loader, c, args, progress=True, prefix="Val:",
                    mask_u=mask_activation_noise, masked_features='masked' in args.evaluate)
        if 'test' in args.evaluate:
            print('eval on test data')
            test_loader = DataLoader(test_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=True, 
                pin_memory=True, persistent_workers=te_pw)
            test_acc_1, test_acc_5, test_macro_acc = test_knn(model, feauture_bank, feature_labels, test_loader, c, args, progress=True, prefix="Test:",
                    mask_u=mask_activation_noise, masked_features='masked' in args.evaluate)
        exit()

    if args.evaluate is not None:
        print(f"Starting evaluation name: {args.name}")
        if len(args.evaluate) == 0:
            args.evaluate = ['train', 'val', 'test']

        mask_activation_noise = model.module.mask_fun.sample().detach()

        if 'train' in args.evaluate:
            print('eval on train data')
            transform = train_transform if 'train' in args.train_transform else test_transform
            train_data  = utils.Imagenet(root=args.data + '/train', transform=transform, target_transform=target_transform, class_to_idx=class_to_idx)
            train_loader = DataLoader(train_data, batch_size=tr_bs, num_workers=tr_nw, prefetch_factor=tr_pf, shuffle=False, 
               pin_memory=True, persistent_workers=tr_pw)
            train_acc_1, train_acc_5, train_macro_acc = test(model, train_loader, args, num_classes=c, progress=True, prefix="Train:", mask_u=mask_activation_noise)
        if 'val' in args.evaluate:
            print('eval on val data')
            val_loader = DataLoader(val_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
                pin_memory=True, persistent_workers=te_pw)
            val_acc_1, val_acc_5, val_macro_acc = test(model, val_loader, args, num_classes=c, progress=True, prefix="Val:", mask_u=mask_activation_noise)
        if 'test' in args.evaluate:
            print('eval on test data')
            test_loader = DataLoader(test_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
                pin_memory=True, persistent_workers=te_pw)
            test_acc_1, test_acc_5, test_macro_acc = test(model, test_loader, args, num_classes=c, progress=True, prefix="Test:", mask_u=mask_activation_noise)
        exit()

    if not args.resume and os.path.exists(log_file):
        os.remove(log_file)            
    
    if args.gradnorm_rescale_weights:
        gradnorm_balancer.rescale_weights()
    
    kwargs = {'ema': ema, 'gradnorm_balancer': gradnorm_balancer, 'gradnorm_optimizer': gradnorm_optimizer, 'log_file': log_file}
    kwargs.update(losses_and_penalty_dict)
    ssl_type = args.ssl_type.lower()
    if ssl_type == 'moco' or ssl_type == 'mocosupcon':
        kwargs.update({'temperature': temperature})
        if ssl_type == 'mocosupcon':
            def filter_indices(idxs, labels, partition, **kwargs):
                idxs = idxs[labels==partition]
                return idxs
            kwargs.update({'filter_indices': filter_indices})
    elif ssl_type == 'simsiam':
        pass
        
    if args.loss_CE_type == 'CEweighted': # weight per-class loss w/ its inverse frequency
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
    
    memory_hash = utils.compute_dataset_fingerprint(memory_data)
    if args.only_cluster or not os.path.exists(fp): # recalculate cluster OR cluster doesn't exist
        memory_loader = DataLoader(memory_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
            pin_memory=False, persistent_workers=te_pw)
        # Cannot use end="" b/c cal_cosine_distance prints progress bar and overwrites its
        if args.only_cluster:
            print('Recalculation of cluster file requested... ')
        else:
            print('No cluster file, creating... ')
        if args.cluster_save_dist: # save cluster distances
            env_ref_set, partitions, dist = utils_cluster.cal_cosine_distance(model, memory_loader, args.class_num, temperature=args.cluster_temp, 
                anchor_class=None, class_debias_logits=True, return_dist=True, K=args.num_clusters)
            os.makedirs(os.path.dirname(fp_dist), exist_ok=True)
            # dist is a dictionary with anchor classes as keys of similarity scores
            torch.save(dist, fp_dist)
            print(f"Cluster distances saved in {fp_dist}")
        else:
            env_ref_set, partitions = utils_cluster.cal_cosine_distance(model, memory_loader, args.class_num, temperature=args.cluster_temp, 
                anchor_class=None, class_debias_logits=True, K=args.num_clusters)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        torch.save({'partitions': partitions, 'memory_hash': memory_hash}, fp)
        print(f'cluster {fp} ready!') 
        if args.only_cluster:
            exit(1)
        
        memory_loader = shutdown_loader(memory_loader)
        gc.collect()              # run Python's garbage collector
    else:
        checkpoint = torch.load(fp)
        partitions = checkpoint.get("partitions", None)
        assert partitions is not None, f"No partitions in cluster file {fp}"
        memory_hash_ = checkpoint.get("memory_hash", None)
        
        if memory_hash is not None and memory_hash_ is not None:
            assert memory_hash == memory_hash_, f"Current train dataset hash {memory_hash} != hash from checkpoint {memory_hash_}!" 

        print(f'Cluster {fp} loaded.')
        assert len(partitions) == args.num_clusters, "Num clusters in cluster file {} != num_clusters {}".format(len(partitions), args.num_clusters)
        assert partitions[0].size(1) == args.env_num, "Num envs in cluster file {} != num_envs {}".format(partitions[0].size(1), args.env_num)
        assert args.clusters_to_use is None or \
            max(args.clusters_to_use) <= args.num_clusters-1, "Largest cluster to use {} must be < {}".format(max(args.clusters_to_use), args.num_clusters)
    partitions = [p.to(device) for p in partitions]

    # 'partitions' should be a list of splits, each one the size of the whole dataset w/ dim=1 equal to the number of environments.
    # each entry is a weight of sample's membership "strength" in an environment

    train_loader = None

    print("Running training with args:")
    print(args)
    print() # insert separating line
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

        mask_activation_noise = model.module.mask_fun.sample().detach()

        if (epoch >= args.test_freq) and ((epoch % args.test_freq == 0) or (epoch == epochs)): # eval model every test_freq epochs
            test_loader = DataLoader(test_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
                pin_memory=False, persistent_workers=te_pw)
            test_acc_1, test_acc_5, test_macro_acc = test(model, test_loader, args, num_classes=c, progress=True, prefix="Test:", mask_u=mask_activation_noise)
            test_loader = shutdown_loader(test_loader)
            gc.collect()              # run Python's garbage collector
            """
            txt_write = open("results-eqinv/{}/{}/{}".format(args.dataset, args.name, 'inference_result.txt'), 'a')
            txt_write.write('\ntest_acc@1: {}, test_acc@5: {}, test_macro_acc: {}'.format(test_acc_1, test_acc_5, test_macro_acc))
            torch.save(model.state_dict(), 'results-eqinv/{}/{}/model_{}.pth'.format(args.dataset, args.name, epoch))
            """

        if (epoch >= args.val_freq) and ((epoch % args.val_freq == 0) or (epoch == epochs)) and (args.dataset == 'ImageNet'):
            # evaluate on validation set
            val_loader = DataLoader(val_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
                pin_memory=False, persistent_workers=te_pw)
            acc1, _, _ = test(model, val_loader, args, num_classes=c, progress=True, prefix="Val:", mask_u=mask_activation_noise)
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
                'queue_proj':           queue_proj,
                'queue_noproj':         queue_noproj,
                'state_dict_gradnorm':  gradnorm_balancer.state_dict(),
                'gradnorm_optimizer':   gradnorm_optimizer.state_dict(),
                "rng_dict": {
                    "rng_state":        torch.get_rng_state(),
                    "cuda_rng_state":   cuda_rng_state,
                    "numpy_rng_state":  np.random.get_state(),
                    "python_rng_state": random.getstate(),
                },
                'ema':                  ema,
                'args':                 args,
            }, is_best, filename='{}/{}/checkpoint.pth.tar'.format(args.save_root, args.name))

        if args.opt_mask and args.mask_save_freq and ((epoch % args.mask_save_freq == 0) or (epoch == epochs)):
            torch.save(model.module.mask_fun.mask, '{}/{}/mask_layer_opt'.format(args.save_root, args.name))
