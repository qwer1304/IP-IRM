import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import random
import shutil
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

import utils
from model import ModelResnet, SimSiam
from prepare import prepare_datasets, traverse_objects
import gc
from math import ceil, prod
import copy
import traceback
import sys
import time

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

class FeatureQueue:
    def __init__(self, queue_size, dim, device=None, dtype=torch.float32):
        """
        A circular queue for storing feature embeddings.

        Args:
            queue_size: int
                Maximum number of elements in the queue.
            dim: int
                Dimension of each feature vector.
            device: torch.device or None
                Device to store the queue on (default: same as k when first updated).
            dtype: torch.dtype
                Data type of the queue (default: torch.float32).
        """
        self.queue_size = queue_size
        self.dim = dim
        self.device = device
        self.dtype = dtype

        self.queue = F.normalize(torch.randn(queue_size, dim, device=device, dtype=dtype), dim=1)
        self.write_ptr = 0  # write index - first index to write from
        self.read_ptr = 0  # read index - first index to read from 

    @torch.no_grad()
    def update(self, k):
        """
        Update the queue with new keys.

        Args:
            k: torch.Tensor, shape [batch_size, dim]
                New keys to enqueue.
        """
        n = k.size(0)
        if n == 0:
            return

        if self.write_ptr + n <= self.queue_size:
            self.queue[self.write_ptr:self.write_ptr+n] = k
            self.write_ptr = self.write_ptr + n
        else:  # wrap around
            first = self.queue_size - self.write_ptr
            self.queue[self.write_ptr:] = k[:first]
            self.queue[:n-first] = k[first:]
            self.write_ptr = n - first

    def get(self, n=None, advance=True):
        """Return the current queue tensor."""
        # n: if n>0    - number of elements from the current read location to return
        #    if n=None - return the whole queue
        if n is None:
            n = self.queue_size
        else: 
            assert (n <= self.queue_size) and (n > 0)
        
        if self.read_ptr + n <= self.queue_size:
            k = self.queue[self.read_ptr:self.read_ptr+n]
            if advance:
                self.read_ptr = self.read_ptr + n
        else:  # wrap around
            first = self.queue_size - self.read_ptr
            k1 = self.queue[self.read_ptr:]
            k2 = self.queue[:n-first]
            k = torch.cat([k1,k2])
            if advance:
                self.read_ptr = n - first
        return k

def microbatches(x, y, mb_size, min_size=2):
    # yields a micro-batch
    N = x.size(0)
    yb = None
    for i in range(0, N, mb_size):
        xb = x[i:i+mb_size] 
        if y is not None:
            yb = y[i:i+mb_size]
        if xb.size(0) < min_size:
            continue  # skip this tiny batch
        yield xb, yb

class BaseCalculator:
    def __init__(self, loss_module, *args, debug=False, device='cuda', **kwargs):
        self.loss_module         = loss_module
        self.debug               = debug

    def penalty(self, idxs=None, **kwargs):
        raise NotImplementedError
        
    def penalty_finalize(self, grads, szs):
        raise NotImplementedError

    def penalty_grads_finalize(self, grads, penalties, szs):
        raise NotImplementedError

    @staticmethod
    def num_halves():
        raise NotImplementedError

class VRExCalculator(BaseCalculator):
    """
    Class for VREx calculation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def penalty(self, loss, *args, **kwargs):
        return loss
        
    def penalty_finalize(self, penalties, szs, **kwargs):
        """
            penalties:  Penalty per half, per env, unnormalized (1,num_partitions,num_envs)
            szs:        sizes of halves of environments
        """
        mu = (penalties / szs).mean(dim=[0,2], keepdim=True) # (1,num_partitions,1)
        
        return ((penalties / szs - mu)**2) # normalized per env for macro-batch, (1, num_partitions, num_envs)

    def penalty_grads_finalize(self, grads, penalties, szs):
        """
        Given dPenalty/dTheta, Penalty per half, per env and their sizes calculate the combined gradient.
        dV/dTheta = 2/E*sum_e((Loss_e - mu) * grad_e), where mu = 1/E*sum_e(Loss_e) 
            grads:      dPenalty/dTheta per half, per env, unnormalized (1,num_partitions,num_envs,parnums)
            penalties:  Penalty per half, normalized per env, (1,num_partitions,num_envs)
            szs:        sizes of halves of environments
        """
        
        _, num_partitions, num_env    = szs.size()
        mu      = penalties.mean(dim=[0,2], keepdim=True) # (1,num_env,1)
        x       = (2 * (penalties[..., None] - mu[..., None]) 
                     * (grads / szs[..., None]) 
                     / num_env
                  ).sum(dim=(0,1,2)) / num_partitions # (parnums,)
            
        total_grad_flat = x
        return total_grad_flat

    @staticmethod
    def num_halves():
        return 1
        
# ---------------------------
# Base IRM Calculator
# ---------------------------

"""
L = 1/Ntr * (nll + IRM)
nll = sum_e(nll_e)
IRM = sum_e(IRM_e)
nll_e = 1/Ne sum_i(nll_i(f(xi),yi))
IRM_e = g1 * g2 # two halves of Ni 
gi = 1/Ni d/ds sum_j(nll_j(s*f(xj),yj)) = 1/Ni sum_j(d/ds nll_j(s*f(xj),yj))
d/dTheta L = 1/Ntr * (d/dTheta nll + d/dTheta IRM)
d/dTheta nll = 1/Ne sum_j(d/dTheta nll_j(f(xj),yj))
d/dTheta IRM = sum_e(d/dTheta IRM_e)
d/dTheta IRM_e = d/dTheta (g1 * g2) = d/dTheta g1 * g2 + g1 * d/dTheta g2
d/dTheta gi = 1/Ni sum_j(d/dTheta d/ds nll_j(s*f(xj),yj))
"""
        
class IRMCalculator(BaseCalculator):
    """
    Base class for IRM calculation. Subclass this and implement
    `gradients_for_half` to return g_i for a half-batch.
    """
    def __init__(self, *args, irm_temp=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.irm_temp            = irm_temp

    def penalty(self, *args, **kwargs):
        """
        Compute gradient w.r.t. scale for a half-batch.
        Must be implemented by subclass.
        Returns a tensor g_i.
        """
        raise NotImplementedError
        
    def penalty_finalize(self, penalties, szs, keep_halves=False):
        if not keep_halves:
            return (penalties[0] / szs[0]) * (penalties[1] / szs[1])  # normalized per env for macro-batch 
        else:
            penalties[0] /= szs[0]
            penalties[1] /= szs[1]
            return penalties

    def penalty_grads_finalize(self, grads, penalties, szs):
        """
        Given dPenalty/dTheta, Penalty per half, per env and their sizes calculate the combined gradient.
            grads:      dPenalty/dTheta per half, per env, unnormalized
            penalties:  Penalty per half, normalized per env
            szs:        sizes of halves of environments
        """
        # IRM = gs1 * gs2, where gs1 and gs2 are gradients w.r.t. scaler of mean CE of halves of sample in a batch
        # dIRM/dTheta = d(gs1 * gs2)/dTheta = dgs1/dTheta * gs2 + gs1 * dgs2/dTheta

        num_halves, num_partitions, num_env = szs.size()

        for i in range(num_halves):
            j = (i + num_halves + 1) % num_halves
            x = (  (grads[i] / szs[i, ..., None])
                 * penalties[j, ..., None]
                 / num_env 
                ).sum(dim=(0,1)) / num_partitions  # shape (param_numel,)
            if i == 0:
                total_grad_flat = x
            else:
                total_grad_flat += x

        return total_grad_flat

    @staticmethod
    def num_halves():
        return 2

# ---------------------------
# CE-based IRM (for MoCo)
# ---------------------------
class CE_IRMCalculator(IRMCalculator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def penalty(self, *args, idxs=None, **kwargs):
        device = self.loss_module.logits().device
        # one scalar (requires grad)
        batch_size = self.loss_module.logits(idxs=idxs).size(0)
        s = torch.ones(batch_size, device=device, requires_grad=True)  # one s per sample
        # Compute g_i in a CE-specific way

        # scaler (s) multiplies a tensor (B,logits), so need to unsqueeze dim=1
        losses = self.loss_module.compute_loss_micro(idxs=idxs, scale=s.unsqueeze(1), temperature=self.irm_temp, **kwargs)
        grad_outputs = torch.ones(1, losses.size(0), device=device)
        g_i = torch.autograd.grad(
            losses,
            s,
            create_graph=True,  # keep graph for next loss
            grad_outputs=grad_outputs, 
            is_grads_batched=True
        )
        # g_i is a tuple w/ entries corresponding to gradients w.r.t each parameter (here - s)
        # each entry is a tensor w/ dim=0 corresponding to each row in 'grad_outputs'
        # select 1st parameter (s) and squeeze out the dim dimension (which is 1)
        g_i = g_i[0].squeeze(0)
        return g_i

class SimSiamIRMCalculator(IRMCalculator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def penalty(self, losses, **kwargs):
        device = self.loss_module.representations[0].device
        # one scalar (requires grad)
        batch_size = self.loss_module.logits(idxs=idxs).size(0)
        s = torch.ones(batch_size, device=device, requires_grad=True)  # one s per sample
        # Compute g_i in a CE-specific way

        g_i = torch.autograd.grad(
            losses * s, # losses is a tensor of scalars
            s,
            create_graph=True,  # keep graph for next loss
            grad_outputs=grad_outputs, 
            is_grads_batched=True
        )
        # g_i is a tuple w/ entries corresponding to gradients w.r.t each parameter (here - s)
        # each entry is a tensor w/ dim=0 corresponding to each row in 'grad_outputs'
        # select 1st parameter (s) and squeeze out the dim dimension (which is 1)
        g_i = g_i[0].squeeze(0)
        return g_i
        
# ---------------------------
# Base Loss Module
# ---------------------------
class LossModule:
    """
    Base class for pluggable loss module.
    Subclass for MoCo, SimSiam, etc.
    """
    def __init__(self, net, device='cuda', **kwargs):
        self.net = net

    def pre_batch(self, batch_data):
        pass

    def pre_micro_batch(self, batch_data):
        pass

    def compute_loss_micro(self, batch_data):
        raise NotImplementedError

    def post_micro_batch(self):
        pass

    def post_batch(self):
        pass

    def prepare_for_free(self):
        for k, v in list(self.__dict__.items()):
            if isinstance(v, torch.Tensor) and v.is_cuda:
                setattr(self, k, None)    

    def get_debug_info_str(self):
        return ""


    def loss_grads_finalize(self, grads, losses, szs):
        """
            grads:  Penalty per half, unnormalized per env
            losses: Losses per half, normalized per env
            szs:    sizes of halves of environments
        """
        num_env = prod(szs.size())
        total_grad_flat  = (  grads  
                            / szs[..., None] 
                            / num_env
                           ).sum(dim=(0,1,2))        # shape (param_numel,)
        return total_grad_flat

# ---------------------------
# MoCo Loss Module
# ---------------------------
class MoCoLossModule(LossModule):
    def __init__(self, *args, net_momentum=None, queue=None, temperature=None, debug=False, **kwargs):
        super().__init__(*args, **kwargs)
        assert net_momentum is not None
        assert queue is not None
        self.net_momentum = net_momentum
        self.momentum = kwargs['momentum']
        self.net_momentum.train()
        self.queue = queue
        self.temperature = temperature or 1.0
        self.this_batch_size = 0
        self.debug = debug
        if self.debug:
            self.total_pos = 0.0
            self.total_neg = 0.0
            self.total_maxneg = 0.0
            self.count = 0

    def pre_batch(self, batch_data):
        self.this_batch_size = len(batch_data)
        self.queue.get(self.this_batch_size) # advance read pointer

    def pre_micro_batch(self, pos, transform, normalize=True):
        pos_q = transform(pos)
        pos_k = transform(pos)

        _, out_q = self.net(pos_q)
        if normalize:
            out_q = F.normalize(out_q, dim=1)
        with torch.no_grad():
            _, out_k = self.net_momentum(pos_k)
            if normalize:
                out_k = F.normalize(out_k, dim=1)
        self.out_k = out_k # save in state for queue update at end of batch
        
        l_pos = torch.sum(out_q * out_k, dim=1, keepdim=True)
        l_neg = torch.matmul(out_q, self.queue.get((self.queue.queue_size - self.this_batch_size), advance=False).t())
        self._logits = torch.cat([l_pos, l_neg], dim=1)
        self.labels = torch.zeros(self._logits.size(0), dtype=torch.long, device=self._logits.device)
        if self.debug:
            self.total_pos    += l_pos.mean().item() * l_pos.size(0)
            self.total_neg    += l_neg.mean().item() * l_pos.size(0)
            self.total_maxneg += l_neg.max().item()  * l_pos.size(0)
            self.count        += l_pos.size(0)
        
    def logits(self, idxs=None):
        if idxs is None:
            idxs = torch.arange(self._logits.size(0), device=self._logits.device)
        return self._logits[idxs]
        
    def targets(self, idxs=None):
        if idxs is None:
            idxs = torch.arange(self._logits.size(0), device=self._logits.device)
        return self.labels[idxs]

    def compute_loss_micro(self, idxs=None, scale=1.0, temperature=None, reduction='sum'):
        if idxs is None:
            idxs = torch.arange(self._logits.size(0), device=self._logits.device)
        # sum over batch, per env handled by driver
        temperature = temperature or self.temperature
        loss = F.cross_entropy(scale * self._logits[idxs] / temperature, self.labels[idxs], reduction=reduction)
        return loss

    def post_micro_batch(self):
        self.queue.update(self.out_k.detach())

    def post_batch(self):
        with torch.no_grad():
            for param_q, param_k in zip(self.net.parameters(), self.net_momentum.parameters()):
                param_k.mul_(self.momentum).add_(param_q, alpha=1.0 - self.momentum)
        if self.debug:
            self.total_pos = 0.0
            self.total_neg = 0.0
            self.total_maxneg = 0.0
            self.count = 0

    def get_debug_info_str(self):
        if self.debug:
            mean_pos = self.total_pos / self.count
            mean_neg = self.total_neg / self.count
            mean_maxneg = self.total_maxneg / self.count
            return f' mean_pos: {mean_pos:.4f} mean_neg: {mean_neg:.4f} mean_maxneg: {mean_maxneg:.4f}'
        else:
            return ""

# ---------------------------
# SimSiam Loss Module
# ---------------------------
class SimSiamLossModule(LossModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pre_micro_batch(self, pos, transform, normalize=True):
        x1 = transform(pos)
        x2 = transform(pos)

        _, z1 = self.net(x1)
        _, z2 = self.net(x2)
        p1 = self.net.module.predictor(z1)
        p2 = self.net.module.predictor(z2)
        self.representations = (z1, z2, p1, p2)

    def compute_loss_micro(self, idxs=None, scale=1.0, reduction='sum'):
        """
        Computes unnormalized loss of a micro-batch
        """
        z1, z2, p1, p2 = self.representations
        if idxs is None:
            idxs = torch.arange(z1.size(0), device=z1.device)
        # symmetric SimSiam loss (neg cosine, average two directions)
        loss_dir1 = - F.cosine_similarity(scale * p1[idxs], z2[idxs].detach(), dim=-1)
        loss_dir2 = - F.cosine_similarity(scale * p2[idxs], z1[idxs].detach(), dim=-1)
        loss = 0.5 * (loss_dir1 + loss_dir2)
        if reduction == 'sum':
            loss = loss.sum()
        return loss

# ssl training with IP-IRM
def train_env(net, train_loader, train_optimizer, partitions, batch_size, args, **kwargs):
    # Initialize dictionaries to store times

    net.train()
    
    if isinstance(partitions, list): # if retain previous partitions
        assert args.retain_group
    else:
        partitions = [partitions]
    num_partitions = len(partitions)
    
    device = next(net.parameters()).device

    transform = train_loader.dataset.transform
    target_transform = train_loader.dataset.target_transform

    if args.increasing_weight:
        penalty_weight = utils.increasing_weight(args.increasing_weight, args.penalty_weight, args.penalty_iters, epoch, args.epochs)
    elif args.penalty_iters < 200:
        penalty_weight = args.penalty_weight if epoch >= args.penalty_iters else 0.
    else:
        penalty_weight = args.penalty_weight
        
    loss_weight      = args.penalty_cont * (1 if penalty_weight <= 1 else 1 / penalty_weight)
    loss_keep_weight = args.penalty_keep_cont * (1 if penalty_weight <= 1 else (1 / penalty_weight))
    penalty_weight   = 1 if penalty_weight > 1 else penalty_weight
    
    loader_batch_size            = batch_size
    gradients_accumulation_steps = args.gradients_accumulation_batch_size // loader_batch_size 
    gpu_batch_size               = args.micro_batch_size
    gpu_accum_steps              = ceil(loader_batch_size / gpu_batch_size) # better round up 

    gradients_accumulation_step = 0
    total_samples               = len(train_loader.dataset)
    
    trained_samples      = 0
    total_keep_cont_loss = 0.0
    total_cont_loss      = 0.0
    total_irm_loss       = 0.0
    total_loss           = 0.0

    bar_format = '{l_bar}{bar:' + str(args.bar) + '}{r_bar}' #{bar:-' + str(args.bar) + 'b}'
    train_bar = tqdm(train_loader,
            total=len(train_loader),        # number of batches
            ncols=args.ncols,               # total width available
            dynamic_ncols=False,            # disable autosizing
            bar_format=bar_format,          # request bar width
            )

    # instantiate LossModule and IRMCalculator based on args (pluggable)
    # default to MoCo if args.loss_type not provided
    loss_type = getattr(args, 'ssl_type', 'moco')
    loss_type = loss_type.lower()
    penalty_type = getattr(args, 'penalty_type', 'irm')
    penalty_type = penalty_type.lower()

    if loss_type == 'moco':
        LossModule = MoCoLossModule
    elif loss_type == 'simsiam':
        LossModule = SimSiamLossModule
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    loss_module = LossModule(net, debug=args.debug, **kwargs)

    # IRM calculator selection
    if penalty_type == 'irm':
        if loss_type   == 'moco':
            PenaltyCalculator = CE_IRMCalculator
        elif loss_type == 'simsiam':
            PenaltyCalculator = SimSiamIRMCalculator
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
    elif penalty_type == 'vrex':
        PenaltyCalculator = VRExCalculator
    else:
        raise ValueError(f"Unknown penalty_type: {penalty_type}")

    penalty_calculator   = PenaltyCalculator(loss_module, irm_temp=args.irm_temp, debug=args.debug, **kwargs)
    """
    We made an attempt to get rid of halving the micro-batches of the whole batch into two subsets. 
    Turns out this cannot be done because losses and gradients are aggregated over micro-batches, 
    but aggregations over halves are needed for IRM and it's impossible to recover back the halves 
    from full aggregators.
    """
    num_halves  = PenaltyCalculator.num_halves()

    loss_aggregator      = torch.zeros((num_halves, num_partitions, args.env_num), dtype=torch.float, device=device) 
    penalty_aggregator   = torch.zeros((num_halves, num_partitions, args.env_num), dtype=torch.float, device=device) 
    loss_keep_aggregator = torch.tensor(0, dtype=torch.float, device=device) # scalar
    halves_sz            = torch.zeros((num_halves, num_partitions, args.env_num), dtype=torch.float, device=device) 
    # One buffer per parameter
    loss_grads = [  # dLoss / dTheta
        torch.zeros((*loss_aggregator.shape, p.numel()), dtype=p.dtype, device=p.device)
        for p in net.parameters()
    ]
    penalty_grads = [ # dPenalty / dTheta
        torch.zeros((*penalty_aggregator.shape, p.numel()), dtype=p.dtype, device=p.device)
        for p in net.parameters()
    ]
    loss_keep_grads = [  # dLoss / dTheta
        torch.zeros(p.numel(), dtype=p.dtype, device=p.device)
        for p in net.parameters()
    ]

    train_optimizer.zero_grad(set_to_none=True) # clear gradients at the beginning 

    for batch_index, data_env in enumerate(train_bar):

        data_batch, indexs_batch = data_env[0], data_env[-1] # 'data_batch' is an batch of images, 'indexs_batch' is their corresponding indices 
        this_batch_size = len(indexs_batch) # for the case drop_last=False
        
        loss_module.pre_batch(data_batch)

        # -----------------------
        # Step 0: micro-batches
        # -----------------------
        mb_list = list(microbatches(data_batch, indexs_batch, gpu_batch_size))

        for j in range(num_halves): # over halves of micro-batches
            for i in [i_ for i_ in range(len(mb_list)) if i_ % num_halves == j]: # loop over micro-batches
                # per micro-batch pipeline
                batch_micro, indexs = mb_list[i]
                batch_micro         = batch_micro.cuda(non_blocking=True)
                indexs              = indexs.cuda(non_blocking=True)

                num_samples           = len(batch_micro)
                num_split_repeates    = int(not args.baseline) * (int(loss_weight>0) + int(penalty_weight>0))
                num_baseline_repeates = int(loss_keep_weight>0) * int(args.keep_cont)                                  
                num_repeats           = max(num_split_repeates, num_baseline_repeates)
                num_grads             = num_partitions * args.env_num * num_split_repeates + num_baseline_repeates
                grad_outputs          = torch.zeros((num_grads, num_samples*num_repeats), dtype=torch.float, device=device) 
                differentiate_this    = []

                """
                prepare for micro-batch in loss-sepcific way:
                    MoCo:    generate two views, get their embeddings from respective encoders, normalize them, etc
                    SimSiam: generate two views, get their projections and predictions, etc
                """
                loss_module.pre_micro_batch(batch_micro, transform=transform, normalize=True)

                # -----------------------
                # SSL
                # -----------------------

                # compute unnormalized micro-batch loss
                losses_samples = loss_module.compute_loss_micro(reduction='none')
                if (loss_weight > 0) or (args.keep_cont and (loss_keep_weight > 0)):
                    differentiate_this.append(losses_samples)
                if penalty_weight > 0:
                    penalties_samples = penalty_calculator.penalty(losses_samples, reduction='none')
                    differentiate_this.append(penalties_samples)

                if not args.baseline:
                    for partition_num, partition in enumerate(partitions):
                        for env in range(args.env_num):

                            # split mb: 'idxs' are indices into 'indexs' that correspond to domain 'env' in 'partition'
                            idxs = utils.assign_idxs(indexs, partition, env)

                            if (N := len(idxs)) == 0:
                                continue

                            halves_sz[j,partition_num,env] += N # update number of elements in environment
                            
                            # losses
                            if loss_weight > 0:
                                # compute unnormalized micro-batch loss
                                loss = losses_samples[idxs].sum(dim=0).detach()
                                loss_aggregator[j,partition_num,env] += loss # unnormalized, before penalty scaler
                            if penalty_weight > 0:
                                penalty = penalties_samples[idxs].sum(dim=0).detach()
                                penalty_aggregator[j,partition_num,env] += penalty # unnormalized penalty components before penalty scaler

                            # gradients
                            linear_idx = torch.tensor(partition_num*args.env_num + env, dtype=torch.int, device=device)
                            offset = 0
                            mask = torch.zeros(num_samples, dtype=torch.float, device=device)
                            mask[idxs] = 1.0
                            if loss_weight>0:
                                grad_outputs[linear_idx][offset:offset+num_samples] = mask * loss_weight
                                linear_idx += num_partitions * args.env_num
                                offset += num_samples
                            if penalty_weight>0:
                                grad_outputs[linear_idx][offset:offset+num_samples] = mask * penalty_weight
                                offset += num_samples
                        # end for env in range(args.env_num):
                    # end for partition_num, partition in enumerate(partitions):
                # end if not args.baseline:

                if args.keep_cont and (loss_keep_weight > 0): # global loss @ 1st partition
                    # This could be done w/o the split into two halves, but this streamlines the code w/o any harm
                    # Here we know that losses are over the whole macro-batch, so we can normalize up-front
                    loss = losses_samples.sum().detach()  / this_batch_size / gradients_accumulation_steps
                    # compute unnormalized gradients for this loss
                    # grad_outputs: one per sample
                    loss_keep_aggregator += loss # before scaler

                    offset = 0 # use losses
                    grad_outputs[-1][offset:offset+num_samples]  = 1.0 * loss_keep_weight / this_batch_size / gradients_accumulation_steps
                    # don't need to add to losses to be differentiated b/c it uses the same losses
                    # differentiate_this.append(losses_samples)

                differentiate_this = torch.cat(differentiate_this, dim=0)

                # compute all needed grads
                # 'grads_all' is a tuple w/ an entry per parameter.
                # each entry is a tensor w/ 1st dim = 'grad_outputs.size(0)' and other dims matching the parameter

                """
                print()
                print(f"num_samples {num_samples}, num_split_repeates {num_split_repeates}, num_baseline_repeates {num_baseline_repeates}," +                                  
                      f"num_repeats {num_repeats}, num_grads {num_grads}, grad_outputs {grad_outputs.size()}, differentiate_this {differentiate_this.size()}")
                """

                grads_all = torch.autograd.grad(
                    differentiate_this,
                    tuple(net.parameters()),
                    retain_graph=False,  # no need to keep graph for next loss
                    allow_unused=True,
                    grad_outputs=grad_outputs, 
                    is_grads_batched=True
                )

                if args.keep_cont and (loss_keep_weight > 0): # global loss @ 1st partition
                    # 'grads_all' is a tuple w/ an entry per parameter.
                    # each entry is a tensor w/ 1st dim = 'grad_outputs.size(0)' and other dims matching the parameter


                    # flatten and accumulate per parameter
                    # 'grads_all' is a tuple w/ an entry per parameter.
                    # each entry is a tensor w/ 1st dim = 'grad_outputs.size(0)' and other dims matching the parameter
                    for _j, g in enumerate(grads_all):
                        if g is None:
                            continue
                        grads = g[-1] # loss keep is always last
                        if grads is None:
                            continue
                        grads = grads.detach().view(-1)
                        loss_keep_grads[_j] += grads

                if not args.baseline:
                    for _split in range((num_grads - num_baseline_repeates) // max(1,num_split_repeates)):
                        partition_num, env = _split // args.env_num, _split % args.env_num 
                        linear_idx = _split
                        if loss_weight > 0:
                            # flatten and accumulate per parameter
                            # 'grads_all' is a tuple w/ an entry per parameter.
                            # each entry is a tensor w/ 1st dim = 'grad_outputs.size(0)' and other dims matching the parameter
                            for _j, g in enumerate(grads_all):
                                if g is None:
                                    continue
                                grads = g[linear_idx]
                                if grads is None:
                                    continue
                                grads = grads.detach().view(-1)
                                loss_grads[_j][j,partition_num,env] += grads
                            linear_idx += num_partitions * args.env_num # prepare for penalty grads
                        # penalty
                        if penalty_weight > 0:
                            # flatten and accumulate per parameter
                            for _j, g in enumerate(grads_all):
                                if g is None:
                                    continue
                                grads = g[linear_idx]
                                if grads is None:
                                    continue
                                grads = grads.detach().view(-1)
                                penalty_grads[_j][j,partition_num,env] += grads
                # end if not args.baseline:
                loss_module.post_micro_batch()
                loss_module.prepare_for_free()
                
                # free memory of micro-batch
                del batch_micro, indexs, losses_samples, grads, g, grads_all, differentiate_this
                if (loss_weight > 0) or (args.keep_cont and (loss_keep_weight > 0)):
                    del loss
                if loss_weight > 0:
                    pass
                if penalty_weight > 0:
                    del penalties_samples, penalty
            # end for i in [i_ for i_ in range(len(mb_list)) if i_ % 2 == j]:
            torch.cuda.empty_cache()
        # end for j in range(idxs):
        torch.cuda.empty_cache()

        trained_samples += this_batch_size # total number of samples processed so far

        gradients_accumulation_step += 1
        if gradients_accumulation_step < gradients_accumulation_steps:
            continue

        if loss_weight > 0:
            partition_sz = halves_sz.sum(dim=0, keepdim=True) # (1,J,K) # sizes of envs in macro-batch
            loss_env = loss_aggregator.sum(dim=0, keepdim=True) / partition_sz  # per env for macro-batch, normalized per env
        else:
            loss_env = torch.tensor(0, dtype=torch.float)
        if penalty_weight > 0:
            penalty_env = penalty_calculator.penalty_finalize(penalty_aggregator, halves_sz) # normalized per env for macro-batch
        else:
            penalty_env = torch.tensor(0, dtype=torch.float)

        # ema
        """
        ema_data = {'loss_env': loss_env.mean(), 'penalty_env': penalty_env.mean(), 'loss_keep': loss_keep_aggregator.mean()}
        emas = ema.update(ema_data)
        """

        # normalize gradient norms - norms inlude multiplication by their respective scaler

        # Orginal gradients already normalized
        if args.keep_cont and (loss_keep_weight>0):
            for pind, p in enumerate(net.parameters()):
                total_grad_flat  = loss_keep_grads[pind]    # dCont/dTheta, shape (param_numel,)
                if p.grad is None:
                    p.grad   = total_grad_flat.view(p.shape)
                else:
                    p.grad  += total_grad_flat.view(p.shape) # reshape back to parameter shape
        loss_keep_grads_flat = torch.cat([g.detach().clone() for g in loss_keep_grads if g is not None])
        loss_keep_grad_norm = loss_keep_grads_flat.norm() # can be 0
        
        # Environments gradients
        if loss_weight > 0:
            loss_grads_flat = []
            for pind, p in enumerate(net.parameters()):
                dLoss_dTheta_env = loss_grads[pind]     # per env sum of dCont/dTheta, shape (I,J,K,param_numel)
                total_grad_flat  = loss_module.loss_grads_finalize(dLoss_dTheta_env, loss_env, halves_sz)
                if p.grad is None:
                    p.grad   = total_grad_flat.view(p.shape)
                else:
                    p.grad  += total_grad_flat.view(p.shape) # reshape back to parameter shape
                loss_grads_flat.append(total_grad_flat)
            loss_grads_flat = torch.cat([g.detach().clone() for g in loss_grads_flat if g is not None])
            loss_grad_norm = loss_grads_flat.norm()
        else:
            loss_grad_norm = torch.tensor(0., dtype=torch.float, device=device)

        if penalty_weight > 0:
            penalty_grads_flat = []
            penalty_env = penalty_calculator.penalty_finalize(penalty_aggregator, halves_sz) # normalized per env for macro-batch
            for pind in range(len(penalty_grads)):
                dPenalty_dTheta_env = penalty_grads[pind]  # per env sum of dPenalty/dTheta over macro-batch per parameter, shape (I,J,K,param_numel)
                total_grad_flat     = \
                    penalty_calculator.penalty_grads_finalize(
                        dPenalty_dTheta_env, 
                        penalty_calculator.penalty_finalize(penalty_aggregator, halves_sz, keep_halves=True), 
                        halves_sz
                    )              
                penalty_grads_flat.append(total_grad_flat.detach().clone())
            p_grads_flat = torch.cat([g.detach().clone() for g in penalty_grads_flat if g is not None])           
            penalty_grad_norm = p_grads_flat.norm()
            grad_norm_ratio = (loss_keep_grad_norm + loss_grad_norm) / (penalty_grad_norm + 1e-12)

        if ((loss_weight>0) or (args.keep_cont and (loss_keep_weight>0))) and (penalty_weight>0):
            dot = (loss_keep_grads_flat + loss_grads_flat).dot(p_grads_flat)
            cosine = torch.nn.functional.cosine_similarity((loss_keep_grads_flat + loss_grads_flat), p_grads_flat, dim=0)           
        else:
            dot, cosine = torch.tensor(0, dtype=torch.float), torch.tensor(0, dtype=torch.float)

        loss_grad_norm_sq = loss_grad_norm ** 2 + loss_keep_grad_norm ** 2
        penalty_grad_norm_sq = penalty_grad_norm ** 2
        S1 = loss_grad_norm_sq / (dot.abs() + 1e-30)
        S2 = dot.abs() / (penalty_grad_norm_sq + 1e-30)
        
        penalty_grad_scaler = torch.tensor(1., dtype=torch.float, device=device) # default
        if args.scale_penalty_grad and (dot < 0):
            if S2 <= S1:
                penalty_grad_scaler = (S1 + S2) / 2
                    
        # Penalty and its gradients
        if penalty_weight > 0:
            for pind, p in enumerate(net.parameters()):
                if p.grad is None:
                    p.grad   = penalty_grads_flat[pind].view(p.shape) * penalty_grad_scaler
                else:
                    p.grad  += penalty_grads_flat[pind].view(p.shape) * penalty_grad_scaler # reshape back to parameter shape

        dot, cosine, loss_keep_grad_norm, loss_grad_norm, penalty_grad_norm, penalty_grad_scaler, loss_grad_norm_sq, penalty_grad_norm_sq = \
                    dot.item(), cosine.item(), loss_keep_grad_norm.item(), loss_grad_norm.item(), \
                    penalty_grad_norm.item(), penalty_grad_scaler.item(), loss_grad_norm_sq.item(), penalty_grad_norm_sq.item()

        loss_batch = ((loss_keep_weight * loss_keep_aggregator) + # loss_keep_aggregator is a scalar normalized over macro-batch
                      (penalty_weight   * penalty_env.mean())   + # mean over envs normalized over macro-batch
                      (loss_weight      * loss_env.mean())        # mean over envs normalized over macro-batch
                     )

        # -----------------------
        # Step 3: optimizer step
        # -----------------------
        if (args.penalty_iters > 0) and (epoch == args.penalty_iters) and (penalty_weight > 0) and (not args.increasing_weight):
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.

            if args.opt == "Adam":
                train_optimizer = torch.optim.Adam(
                    net.parameters(),
                    lr=train_optimizer.param_groups[0]["lr"],
                    weight_decay=train_optimizer.param_groups[0]["weight_decay"])
            elif args.opt == 'SGD':
                    optimizer = optim.SGD(
                    net.parameters(),
                    lr=train_optimizer.param_groups[0]["lr"],
                    weight_decay=train_optimizer.param_groups[0]["weight_decay"],
                    momentum=args.train_optimizer.param_groups[0]["momentum"])

        train_optimizer.step()
        train_optimizer.zero_grad(set_to_none=True)  # clear gradients at beginning of next gradients batch

        # total loss is sum of losses so far over entire batch aggregation period.
        total_keep_cont_loss += (loss_keep_weight * loss_keep_aggregator).item() * this_batch_size * gradients_accumulation_steps
        total_irm_loss       += (penalty_weight   * penalty_env.mean()).item()   * this_batch_size * gradients_accumulation_steps
        total_cont_loss      += (loss_weight      * loss_env.mean()).item()      * this_batch_size * gradients_accumulation_steps
        total_loss           += loss_batch.item()                                * this_batch_size * gradients_accumulation_steps

        desc_str = f'Train Epoch: [{epoch}/{epochs}] [{trained_samples}/{total_samples}]' + \
                   f' {args.ssl_type}:' + \
                   f' Total: {total_loss/trained_samples:.4f}' + \
                   f' First: {total_keep_cont_loss/trained_samples:.4f}' + \
                   f' Env: {total_cont_loss/trained_samples:.4f}' + \
                   f' {args.penalty_type}: {total_irm_loss/trained_samples:.4g}' + \
                   f' LR: {train_optimizer.param_groups[0]["lr"]:.4f} PW {penalty_weight:.4f}' + \
                   f' dot: {dot:.4g}, cos: {cosine:.4f}, ng_l^2: {loss_grad_norm_sq:.4g} ng_p^2: {penalty_grad_norm_sq:.4g}' + \
                   f' gp_sc: {penalty_grad_scaler:.4f}'
        desc_str += loss_module.get_debug_info_str()
        train_bar.set_description(desc_str)

        if batch_index % 10 == 0:
            utils.write_log('Train Epoch: [{:d}/{:d}] [{:d}/{:d}] {args.ssl_type}: Total: {:.4f} First: {:.4f} Env: {:.4f}'
                            .format(epoch, epochs, trained_samples, total_samples,
                                    total_loss/trained_samples, total_keep_cont_loss/trained_samples, 
                                    total_cont_loss/trained_samples) + 
                            ' {args.penalty_type}: {:.4g} LR: {:.4f} PW {:.4f} dot {:.4g} cos {:.4f} ng_l^2: {:.4g} ng_p^2: {:.4g} gp_sc{:.4f}'
                            .format(total_irm_loss/trained_samples, train_optimizer.param_groups[0]['lr'], penalty_weight, dot, cosine, 
                                    loss_grad_norm_sq, penalty_grad_norm_sq, penalty_grad_scaler), 
                            log_file=log_file)
                                        
        # Prepare for next iteration
        gradients_accumulation_step = 0
        penalty_aggregator.zero_()
        loss_keep_aggregator.zero_()
        loss_aggregator.zero_()
        halves_sz.zero_()
        for par in loss_grads: # over list
            par.zero_()
        for par in penalty_grads: # over list
            par.zero_()
        for par in loss_keep_grads: # over list
            par.zero_()
        del penalty_env, loss_env, loss_batch
        if (penalty_weight > 0) or (loss_weight > 0):
            del total_grad_flat
        if penalty_weight > 0:
            del dPenalty_dTheta_env, penalty_grads_flat, p_grads_flat
        if loss_weight > 0:
            dLoss_dTheta_env, loss_grads_flat
        torch.cuda.empty_cache()

        loss_module.post_batch()
    # end for batch_index, data_env in enumerate(train_bar):
    return total_loss / trained_samples

def train_update_split(net, update_loader, soft_split, random_init=False, args=None):
    utils.write_log('Start Maximizing ...', log_file, print_=True)
    
    transform = update_loader.dataset.transform
    target_transform = update_loader.dataset.target_transform
    
    if random_init:
        utils.write_log('Give a Random Split:', log_file, print_=True)
        soft_split = torch.randn(soft_split.size(), requires_grad=True, device="cuda")
        utils.write_log('%s' %(utils.pretty_tensor_str(soft_split[:3])), log_file, print_=True)
    else:
        utils.write_log('Use Previous Split:', log_file, print_=True)
        soft_split = soft_split.requires_grad_()
        utils.write_log('%s' %(utils.pretty_tensor_str(soft_split[:3])), log_file, print_=True)

    if args.offline: # Maximize Step offline, first extract image features
        net.eval()
        feature_bank_1, feature_bank_2 = [], []
        with torch.no_grad():
            # generate feature bank
            bar_format = '{l_bar}{bar:' + str(args.bar) + '}{r_bar}' #{bar:-' + str(args.bar) + 'b}'
            train_bar = tqdm(update_loader,
                total=len(update_loader),
                ncols=args.ncols,               # total width available
                dynamic_ncols=False,            # disable autosizing
                bar_format=bar_format,          # request bar width
                desc='train_update_split(): Feature extracting'
            )
            for pos_, target, Index in train_bar:
                pos_ = pos_.cuda(non_blocking=True)
      
                if transform is not None:
                    pos_1 = transform(pos_)
                    pos_2 = transform(pos_)
                if target_transform is not None:
                    target = target_transform(target)
                
                feature_1, out_1 = net(pos_1)
                feature_2, out_2 = net(pos_2)
                feature_bank_1.append(out_1.cpu())
                feature_bank_2.append(out_2.cpu())
        feature1 = torch.cat(feature_bank_1, 0)
        feature2 = torch.cat(feature_bank_2, 0)
        updated_split = utils.auto_split_offline(feature1, feature2, soft_split, temperature, args.irm_temp, loss_mode='v2', irm_mode=args.irm_mode,
                                         irm_weight=args.irm_weight_maxim, constrain=args.constrain, cons_relax=args.constrain_relax, nonorm=args.nonorm, 
                                         log_file=log_file, batch_size=uo_bs, num_workers=uo_nw, prefetch_factor=uo_pf, persistent_workers=uo_pw)
    else:
        updated_split = utils.auto_split(net, update_loader, soft_split, temperature, args.irm_temp, loss_mode='v2', irm_mode=args.irm_mode,
                                     irm_weight=args.irm_weight_maxim, constrain=args.constrain, cons_relax=args.constrain_relax, 
                                     nonorm=args.nonorm, log_file=log_file)
    np.save("results/{}/{}/{}_{}{}".format(args.dataset, args.name, 'GroupResults', epoch, ".txt"), updated_split.cpu().numpy())
    return updated_split

def get_feature_bank(net, memory_data_loader, args, progress=False, prefix="Test:"):
    net.eval()
    
    transform = memory_data_loader.dataset.transform
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
        for data, _ in feature_bar:
            data = data.cuda(non_blocking=True)

            if transform is not None:
                data = transform(data)
                
            feature, out = net(data)
            feature_bank.append(feature)
        #end for data, _, _ in feature_bar:

        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous() # places feature_bank on cuda
        # [N]
        dataset = memory_data_loader.dataset
        if hasattr(dataset, "labels"):
            labels = dataset.labels
        else:
            if dataset.target_transform is not None:
                labels = [dataset.target_transform(t) for t in dataset.targets]
            else:
                labels = dataset.targets        
        feature_labels = torch.tensor(labels, device=feature_bank.device)

    return feature_bank, feature_labels

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, feature_bank, feature_labels, test_data_loader, args, progress=False, prefix="Test:"):
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
    
        transform = test_data_loader.dataset.transform
        target_transform = test_data_loader.dataset.target_transform
    
        if args.extract_features:
            test_data_loader.dataset.target_transform = None

        feature_list = []
        pred_labels_list = []
        pred_scores_list = []
        target_list = []
        target_raw_list = []

        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            
            if transform is not None:
                data = transform(data)

            target_raw = target
            if args.extract_features and target_transform is not None:
                target = target_transform(target_raw).cuda(non_blocking=True)

            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank) # places sim_matrix on cuda
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            if progress:
                test_bar.set_description('KNN {} Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                         .format(prefix, epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

            # compute output
            if args.extract_features:
                feature_list.append(feature)
                target_list.append(target)
                target_raw_list.append(target_raw)
                pred_labels_list.append(pred_labels)
                pred_scores_list.append(pred_scores)

        # end for data, _, target in test_bar

        if feature_list:
            feature = torch.cat(feature_list, dim=0)
            target = torch.cat(target_list, dim=0)
            target_raw = torch.cat(target_raw_list, dim=0)
            pred_labels = torch.cat(pred_labels_list, dim=0)
            pred_scores = torch.cat(pred_scores_list, dim=0)

            # Save to file
            prefix = "test" if "Test" in prefix else "val"
            directory = f'results/{args.dataset}/{args.name}'
            fp = os.path.join(directory, f"{prefix}_features_dump.pt")       
            os.makedirs(os.path.dirname(fp), exist_ok=True)

            state = {
                'features':     feature,
                'labels':       target,
                'labels_raw':   target_raw,
                'pred_labels':  pred_labels,
                'pred_scores':  pred_scores,
                'model_epoch':  epoch,
                'n_classes':    args.class_num,
            }

            utils.atomic_save(state, False, filename=fp)
            print(f"Dumped features into {fp}")

    return total_top1 / total_num * 100, total_top5 / total_num * 100
    
def load_checkpoint(path, model, model_momentum, optimizer, device="cuda"):
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

    # Restore momentum model if applicable
    queue = None
    if model_momentum is not None:
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
    else:
        msg_momentum = "momentum encoder not used"
        queue = None

    # Restore optimizer (if available)
    if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Move optimizer tensors to the correct device
        for state in optimizer.state.values():
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

    print("<= loaded checkpoint '{}' (epoch {})".format(path, checkpoint.get("epoch", -1)))

    return model, model_momentum, optimizer, queue, start_epoch, best_acc1, best_epoch, updated_split, updated_split_all, ema

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--ssl_type', default='MoCo', type=str, choices=['MoCo', 'SimSiam'], help='SSL type')    
    parser.add_argument('--penalty_type', default='IRM', type=str, choices=['IRM', 'VREx'], help='Penalty type')        
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--tau_plus', default=0.1, type=float, help='Positive class priorx')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
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
    parser.add_argument('--micro_batch_size', default=32, type=int, help='batch size on gpu')
    parser.add_argument('--gradients_accumulation_batch_size', default=256, type=int, help='batch size of gradients accumulation')
    parser.add_argument('--queue_size', default=10000, type=int, help='momentum model queue size')
    parser.add_argument('--momentum', default=0.995, type=float, help='momentum model momentum')
    parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--debiased', default=False, type=bool, help='Debiased contrastive loss or standard loss')
    parser.add_argument('--dataset', type=str, default='STL', choices=['STL', 'CIFAR10', 'CIFAR100', 'ImageNet'], help='experiment dataset')
    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('--name', type=str, default='None', help='experiment name')
    parser.add_argument('--pretrain_model', default=None, type=str, help='pretrain model used?')
    parser.add_argument('--baseline', action="store_true", default=False, help='SSL baseline?')
    parser.add_argument('--train_envs', type=str, nargs='+', default=None, required=True)
    parser.add_argument('--test_envs', type=str, nargs='+', default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=0)

    #### ours model param ####
    parser.add_argument('--ours_mode', default='w', type=str, help='what mode to use')
    parser.add_argument('--penalty_weight', default=1.0, type=float, help='penalty weight')
    parser.add_argument('--penalty_cont', default=1.0, type=float, help='cont penalty weight')
    parser.add_argument('--penalty_keep_cont', default=1.0, type=float, help='cont keep penalty weight')
    parser.add_argument('--penalty_iters', default=0, type=int, help='penalty weight start iteration')
    parser.add_argument('--increasing_weight', nargs=5, type=float, default=None, help='increasing penalty weight', 
            metavar='penalty_warmup, scale, speed, eps, debug')
    parser.add_argument('--env_num', default=2, type=int, help='num of the environments')

    parser.add_argument('--maximize_iter', default=30, type=int, help='when maximize iteration')
    parser.add_argument('--irm_mode', default='v1', type=str, help='irm mode when maximizing')
    parser.add_argument('--irm_weight_maxim', default=1, type=float, help='irm weight in maximizing')
    parser.add_argument('--irm_temp', default=0.5, type=float, help='irm loss temperature')
    parser.add_argument('--random_init', action="store_true", default=False, help='random initialization before every time update?')
    parser.add_argument('--constrain', action="store_true", default=False, help='make num of 2 group samples similar?')
    parser.add_argument('--constrain_relax', action="store_true", default=False, help='relax the constrain?')
    parser.add_argument('--retain_group', action="store_true", default=False, help='retain the previous group assignments?')
    parser.add_argument('--debug', action="store_true", default=False, help='debug?')
    parser.add_argument('--nonorm', action="store_true", default=False, help='not use norm for contrastive loss when maximizing')
    parser.add_argument('--groupnorm', action="store_true", default=False, help='use group contrastive loss?')
    parser.add_argument('--offline', action="store_true", default=False, help='save feature at the beginning of the maximize?')
    parser.add_argument('--keep_cont', action="store_true", default=False, help='keep original contrastive?')
    parser.add_argument('--pretrain_path', type=str, default=None, help='the path of pretrain model')

    # image
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    # color in label
    parser.add_argument('--target_transform', type=str, default=None, help='a function definition to apply to target')
    parser.add_argument('--class_to_idx', type=str, default=None, help='a function definition to apply to class to obtain it index')
    parser.add_argument('--image_class', choices=['ImageNet', 'STL', 'CIFAR'], default='ImageNet', help='Image class, default=ImageNet')
    parser.add_argument('--class_num', default=1000, type=int, help='num of classes')
    parser.add_argument('--ncols', default=80, type=int, help='number of columns in terminal')
    parser.add_argument('--bar', default=50, type=int, help='length of progess bar')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--save_root', type=str, default='save', help='root dir for saving')
    parser.add_argument('--checkpoint_freq', default=3, type=int, metavar='N',
                    help='checkpoint epoch freqeuncy')
    parser.add_argument('--val_freq', default=3*3, type=int, metavar='N',
                    help='validation epoch freqeuncy')
    parser.add_argument('--test_freq', default=5*5, type=int, metavar='N',
                    help='test epoch freqeuncy')
    parser.add_argument('--norandgray', action="store_true", default=False, help='skip rand gray transform')
    parser.add_argument('--evaluate', action="store_true", default=False, help='only evaluate')
    parser.add_argument('--extract_features', action="store_true", help="extract features for post processiin during evaluate")

    parser.add_argument('--opt', choices=['Adam', 'SGD'], default='Adam', help='Optimizer to use')
    parser.add_argument('--lr', default=0.001, type=float, help='LR')
    parser.add_argument('--SGD_momentum', default=0.9, type=float, help='LR')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay')
    
    parser.add_argument('--ema', action="store_true", help="adjust gradients w/ EMA")
    parser.add_argument('--scale_penalty_grad', action="store_true", help="scale penalty grads")

    # args parse
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.baseline:
        args.penalty_weight, args.penalty_cont = 0, 0
        
    assert ((args.penalty_weight > 0) or (args.penalty_cont > 0)      or  (args.penalty_keep_cont > 0)) or \
           ((args.penalty_cont == 0) and (args.penalty_keep_cont > 0) and (args.penalty_iters == 0))

    # seed
    utils.set_seed(args.seed)

    feature_dim, temperature, tau_plus, k = args.feature_dim, args.temperature, args.tau_plus, args.k
    epochs, debiased,  = args.epochs,  args.debiased
    dl_tr, dl_te, dl_u, dl_uo = args.dl_tr, args.dl_te, args.dl_u, args.dl_uo
    target_transform = eval(args.target_transform) if args.target_transform is not None else None
    class_to_idx = eval(args.class_to_idx) if args.class_to_idx is not None else None
    image_class, image_size = args.image_class, args.image_size

    if not os.path.exists('results/{}/{}'.format(args.dataset, args.name)):
        os.makedirs('results/{}/{}'.format(args.dataset, args.name))
    log_file = 'results/{}/{}/log.txt'.format(args.dataset, args.name)
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
        update_data = utils.STL10_Index(root=args.data, split='train+unlabeled', transform=train_transform, target_transform=target_transform)
        memory_data = utils.STL10(root=args.data, split='train', transform=test_transform, target_transform=target_transform)
        test_data = utils.STL10(root=args.data, split='test', transform=test_transform, target_transform=target_transform)
    elif args.dataset == 'CIFAR10':
        train_transform = utils.make_train_transform(normalize=args.image_class)
        test_transform = utils.make_test_transform(normalize=args.image_class)
        train_data = utils.CIFAR10_Index(root=args.data, train=True, transform=train_transform, target_transform=target_transform, download=True)
        update_data = utils.CIFAR10_Index(root=args.data, train=True, transform=train_transform, target_transform=target_transform)
        memory_data = utils.CIFAR10(root=args.data, train=True, transform=test_transform, target_transform=target_transform)
        test_data = utils.CIFAR10(root=args.data, train=False, transform=test_transform, target_transform=target_transform)
    elif args.dataset == 'CIFAR100':
        train_transform = utils.make_train_transform(normalize=args.image_class)
        test_transform = utils.make_test_transform(normalize=args.image_class)
        train_data = utils.CIFAR100_Index(root=args.data, train=True, transform=train_transform, target_transform=target_transform)
        update_data = utils.CIFAR100_Index(root=args.data, train=True, transform=train_transform, target_transform=target_transform)
        memory_data = utils.CIFAR100(root=args.data, train=True, transform=test_transform, target_transform=target_transform)
        test_data = utils.CIFAR100(root=args.data, train=False, transform=test_transform, target_transform=target_transform)
    elif args.dataset == 'ImageNet':
        train_transform = utils.make_train_transform(image_size, randgray=not args.norandgray, normalize=args.image_class)
        test_transform = utils.make_test_transform(normalize=args.image_class)
        if False:
            wrap = args.extract_features
            # descriptors of train data
            train_desc  =   {'dataset': utils.Imagenet_idx,
                              'transform': train_transform,
                              'target_transform': target_transform,
                              'class_to_index': class_to_idx,
                              'wrap': False, # for changeable target transform
                              'required_split': "in",
                            }
            update_desc =   {'dataset': utils.Imagenet_idx,
                              'transform': train_transform,
                              'target_transform': target_transform,
                              'class_to_index': class_to_idx,
                              'wrap': False, # for changeable target transform
                              'required_split': "in",
                            }
            memory_desc =   {'dataset': utils.Imagenet,
                              'transform': test_transform,
                              'target_transform': target_transform,
                              'class_to_index': class_to_idx,
                              'wrap': False, # for changeable target transform
                              'required_split': "in",
                            }
            val_desc    =   {'dataset': utils.Imagenet,
                              'transform': test_transform,
                              'target_transform': target_transform,
                              'class_to_index': class_to_idx,
                              'wrap': wrap, # for changeable target transform
                              'required_split': "out",
                            }
            # descriptors of test data
            test_desc   =   {'dataset': utils.Imagenet,
                              'transform': test_transform,
                              'target_transform': target_transform,
                              'class_to_index': class_to_idx,
                              'wrap': wrap, # for changeable target transform
                              'required_split': "in",
                            }


            datas = prepare_datasets(args.data, args.train_envs, [train_desc, update_desc, memory_desc, val_desc], args.holdout_fraction, args.seed)
            train_data, update_data, memory_data, val_data = tuple(data[0] for data in datas)

            datas = prepare_datasets(args.data, args.test_envs, [test_desc], 1.0, args.seed)
            test_data = datas[0][0]

            #traverse_objects(update_data)
            #exit()

        else:
            train_data  = utils.Imagenet_idx(root=args.data + '/train', transform=train_transform, target_transform=target_transform, class_to_idx=class_to_idx)
            update_data = utils.Imagenet_idx(root=args.data + '/train', transform=train_transform, target_transform=target_transform, class_to_idx=class_to_idx)
            memory_data = utils.Imagenet(root=args.data     + '/train', transform=test_transform,  target_transform=target_transform, class_to_idx=class_to_idx)
            test_data   = utils.Imagenet(root=args.data     + '/test',  transform=test_transform,  target_transform=target_transform, class_to_idx=class_to_idx)
            val_data    = utils.Imagenet(root=args.data     + '/val',   transform=test_transform,  target_transform=target_transform, class_to_idx=class_to_idx)
        
    # pretrain model
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

    # model setup and optimizer config
    if args.ssl_type.lower() == 'moco':
        model = ModelResnet(feature_dim, image_class=image_class, state_dict=state_dict).cuda()
    elif args.ssl_type.lower() == 'simsiam':
        model = SimSiam(feature_dim, image_class=image_class, state_dict=state_dict).cuda()
    else:
        raise NotImplemented
    if state_dict is not None:
        print("<= loaded pretrained checkpoint '{}'".format(args.pretrain_path))

    model = nn.DataParallel(model)

    if args.ssl_type.lower() == 'moco':
        model_momentum = copy.deepcopy(model)
        for p in model_momentum.parameters():
            p.requires_grad = False
        momentum = args.momentum              # momentum for model_momentum
        queue_size = args.queue_size
        queue = FeatureQueue(queue_size, feature_dim, device=device, dtype=torch.float32)
    elif args.ssl_type.lower() == 'simsiam':
        model_momentum = None
        queue = None
        momentum = None

    if args.opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.SGD_momentum)
    c = len(memory_data.classes) if args.dataset != "ImageNet" else args.class_num
    print('# Classes: {}'.format(c))


    ema = utils.MovingAverage(0.99, oneminusema_correction=True, active=args.ema)

    # optionally resume from a checkpoint
    best_acc1 = 0
    best_epoch = 0
    resumed = False
    if args.resume:
        if os.path.isfile(args.resume):
            (model, model_momentum, optimizer, queue,
             args.start_epoch, best_acc1, best_epoch,
             updated_split, updated_split_all, ema_) = load_checkpoint(args.resume, model, model_momentum, optimizer)
            if ema_ is not None: # exists in checkpoint
                ema = ema_
            ema.set_active(args.ema) # set to what the user has currently set
            # use current LR, not the one from checkpoint
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            resumed = True
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # training loop
    if not os.path.exists('results'):
        os.mkdir('results')

    epoch = args.start_epoch

    if args.evaluate:
        print(f"Staring evaluation name: {args.name}")
        print('eval on val data')
        memory_loader = DataLoader(memory_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
            pin_memory=True, persistent_workers=te_pw)
        feauture_bank, feature_labels = get_feature_bank(model, memory_loader, args, progress=True, prefix="Evaluate:")
        val_loader = DataLoader(val_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=True, 
            pin_memory=True, persistent_workers=te_pw)
        val_acc_1, val_acc_5 = test(model, feauture_bank, feature_labels, val_loader, args, progress=True, prefix="Val:")
        print('eval on test data')
        test_loader = DataLoader(test_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=True, 
            pin_memory=True, persistent_workers=te_pw)
        test_acc_1, test_acc_5 = test(model, feauture_bank, feature_labels, test_loader, args, progress=True, prefix="Test:")
        exit()

    # update partition for the first time, if we need one
    if not args.baseline:
        if (not resumed) or (resumed and (updated_split is None) and ((args.penalty_cont > 0) or (args.penalty_weight > 0))):  
            if args.dataset != "ImageNet":
                updated_split = torch.randn((len(update_data), args.env_num), requires_grad=True, device=device)
            else:
                updated_split = torch.randn((len(update_data), args.env_num), requires_grad=True, device=device)
                if args.offline:
                    upd_loader = DataLoader(update_data, batch_size=u_bs, num_workers=u_nw, prefetch_factor=u_pf, shuffle=False, 
                        drop_last=False, pin_memory=True, persistent_workers=u_pw)
                else:
                    upd_loader = DataLoader(update_data, batch_size=u_bs, num_workers=u_nw, prefetch_factor=u_pf, shuffle=True,
                        drop_last=True, pin_memory=True, persistent_workers=u_pw)
            updated_split = train_update_split(model, upd_loader, updated_split, random_init=args.random_init, args=args)
            updated_split_all = [updated_split.clone().detach()]
            upd_loader = None
            gc.collect()              # run Python's garbage collector

            # Save a baseline checkpoint with initial split to allow skipping its initial creation
            cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

            utils.atomic_save({
                'epoch':                0, # restore is from epoch+1
                'state_dict':           model.state_dict(),
                'best_acc1':            best_acc1,
                'best_epoch':           best_epoch,
                'optimizer':            optimizer.state_dict(),
                'updated_split':        updated_split,
                'updated_split_all':    updated_split_all,
                'state_dict_momentum':  model_momentum.state_dict() if model_momentum else None,
                'queue':                queue,
                "rng_dict": {
                    "rng_state":        torch.get_rng_state(),
                    "cuda_rng_state":   cuda_rng_state,
                    "numpy_rng_state":  np.random.get_state(),
                    "python_rng_state": random.getstate(),
                },
                'ema':                  ema,
            }, False, filename='{}/{}/checkpoint_1st.pth.tar'.format(args.save_root, args.name))

    train_loader = None

    def shutdown_loader(loader):
        """Shutdown and release a DataLoader and its workers immediately."""
        if loader is None:
            return None
        it = getattr(loader, "_iterator", None)
        if it is not None:
            it._shutdown_workers()
        return None

    for epoch in range(args.start_epoch, epochs + 1):
        if train_loader is None:
            train_loader = DataLoader(train_data, batch_size=tr_bs, num_workers=tr_nw, prefetch_factor=tr_pf, shuffle=True, 
                                pin_memory=True, persistent_workers=tr_pw, drop_last=tr_dl)

        # Minimize step
        if not args.baseline:
            upd_split = updated_split_all if args.retain_group else updated_split
        else:
            upd_split = None
            updated_split = None
            updated_split_all = None            

        if args.ssl_type.lower() == 'moco':
            kwargs = {'net_momentum': model_momentum, 'queue': queue, 'temperature': temperature, 'momentum': momentum}
        elif args.ssl_type.lower() == 'simsiam':
            kwargs = {}

        train_loss = train_env(model, train_loader, optimizer, upd_split, tr_bs, args, **kwargs)

        if (epoch % args.maximize_iter == 0) and (not args.baseline):
            # Maximize Step
            train_loader = shutdown_loader(train_loader)
            gc.collect()
            if args.offline:
                upd_loader = DataLoader(update_data, batch_size=u_bs, num_workers=u_nw, prefetch_factor=u_pf, shuffle=False, 
                    pin_memory=False, persistent_workers=u_pw)
            else:
                upd_loader = DataLoader(update_data, batch_size=u_bs, num_workers=u_nw, prefetch_factor=u_pf, shuffle=True, pin_memory=True, 
                    drop_last=True, persistent_workers=u_pw)
            updated_split = train_update_split(model, upd_loader, updated_split, random_init=args.random_init, args=args)
            upd_loader = shutdown_loader(upd_loader)
            gc.collect()              # run Python's garbage collector
            updated_split_all.append(updated_split)

        feature_bank, feature_labels = None, None
        if (epoch % args.test_freq == 0) or \
           ((epoch % args.val_freq == 0) and (args.dataset == 'ImageNet')) or \
           (epoch == epochs): # eval knn every test_freq/val_freq and last epochs
            if train_loader is not None:
                train_loader = shutdown_loader(train_loader)
                gc.collect()
            memory_loader = DataLoader(memory_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
                pin_memory=False, persistent_workers=te_pw)
            feauture_bank, feature_labels = get_feature_bank(model, memory_loader, args, progress=True, prefix="Evaluate:")
            memory_loader = shutdown_loader(memory_loader)
            gc.collect()              # run Python's garbage collector

        if (epoch % args.test_freq == 0) or (epoch == epochs): # eval knn every test_freq epochs
            test_loader = DataLoader(test_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=True, 
                pin_memory=False, persistent_workers=te_pw)
            test_acc_1, test_acc_5 = test(model, feauture_bank, feature_labels, test_loader, args, progress=True, prefix="Test:")
            test_loader = shutdown_loader(test_loader)
            gc.collect()              # run Python's garbage collector
            txt_write = open("results/{}/{}/{}".format(args.dataset, args.name, 'knn_result.txt'), 'a')
            txt_write.write('\ntest_acc@1: {}, test_acc@5: {}'.format(test_acc_1, test_acc_5))
            torch.save(model.state_dict(), 'results/{}/{}/model_{}.pth'.format(args.dataset, args.name, epoch))

        if ((epoch % args.val_freq == 0) or (epoch == epochs)) and (args.dataset == 'ImageNet'):
            # evaluate on validation set
            val_loader = DataLoader(val_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=True, 
                pin_memory=False, persistent_workers=te_pw)
            acc1, _ = test(model, feauture_bank, feature_labels, val_loader, args, progress=True, prefix="Val:")
            val_loader = shutdown_loader(val_loader)
            gc.collect()              # run Python's garbage collector

            # remember best acc@1 & best epoch and save checkpoint
            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
                best_epoch = epoch
        else:
            is_best = False
        if feature_bank is not None:
            feauture_bank, feature_labels = None, None
            gc.collect()              # run Python's garbage collector

        if (epoch % args.checkpoint_freq == 0) or (epoch == epochs):
            cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

            utils.atomic_save({
                'epoch':                epoch,
                'state_dict':           model.state_dict(),
                'best_acc1':            best_acc1,
                'best_epoch':           best_epoch,
                'optimizer':            optimizer.state_dict(),
                'updated_split':        updated_split,
                'updated_split_all':    updated_split_all,
                'state_dict_momentum':  model_momentum.state_dict() if model_momentum else None,
                'queue':                queue,
                "rng_dict": {
                    "rng_state":        torch.get_rng_state(),
                    "cuda_rng_state":   cuda_rng_state,
                    "numpy_rng_state":  np.random.get_state(),
                    "python_rng_state": random.getstate(),
                },
                'ema':                  ema,
            }, is_best, filename='{}/{}/checkpoint.pth.tar'.format(args.save_root, args.name))
