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
from torch.utils.data import Dataset, DataLoader, random_split, Subset

from tqdm.auto import tqdm

import utils
from model import ModelResnet, SimSiam
import gradnorm as gn
from prepare import prepare_datasets, traverse_objects
import gc
from math import ceil, prod
import copy
import traceback
import sys
import time
import warnings
from collections import defaultdict
from typing import Union, List, Dict

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

class FeatureQueue:
    def __init__(self, queue_size, dim, device=None, dtype=torch.float32, indices=False):
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
            indices: bool
                Whether to store keys' indices 
        """
        self.queue_size = queue_size
        self.dim = dim
        self.device = device
        self.dtype = dtype

        self.queue = F.normalize(torch.randn(queue_size, dim, device=device, dtype=dtype), dim=1)
        self.write_ptr = 0  # write index - first index to write from
        self.read_ptr = 0  # read index - first index to read from 
        
        if indices:
            self.indices = torch.randperm(self.queue_size, device=device)
        else:
            self.indices = None

    @torch.no_grad()
    def update(self, k, idx=None):
        """
        Update the queue with new keys.

        Args:
            k:   torch.Tensor, shape [batch_size, dim]
                     New keys to enqueue.
            idx: indices of keys  
        """
        n = k.size(0)
        if n == 0:
            return
        assert ((idx is not None) and (self.indices is not None)) or (idx is None)
        assert (idx is None) or (len(idx) == n), f"idx {idx} n {n}"

        if self.write_ptr + n <= self.queue_size:
            indices = torch.arange(self.write_ptr, self.write_ptr+n)
            self.write_ptr = self.write_ptr + n
        else:  # wrap around
            first = self.queue_size - self.write_ptr
            indices = torch.cat([torch.arange(self.write_ptr, self.queue_size), torch.arange(0, n-first)], dim=0)
            self.write_ptr = n - first
        self.queue[indices] = k
        if idx is not None:
            self.indices[indices] = idx

    def get(self, n=None, advance=True, idx=False):
        """Return the current queue tensor."""
        # n:   if n>0    - number of elements from the current read location to return
        #      if n=None - return the whole queue
        # idx: also return the indices 
        if n is None:
            n = self.queue_size
        else: 
            assert (n <= self.queue_size) and (n > 0)
        assert (idx and (self.indices is not None)) or (not idx), f"idx {idx} self.indices {self.indices}"          
        
        if self.read_ptr + n <= self.queue_size:
            indices = torch.arange(self.read_ptr, self.read_ptr+n)
            if advance:
                self.read_ptr = self.read_ptr + n
        else:  # wrap around
            first = self.queue_size - self.read_ptr
            indices = torch.cat([torch.arange(self.read_ptr, self.queue_size), torch.arange(0, n-first)], dim=0)
            if advance:
                self.read_ptr = n - first
        k = self.queue[indices]
        if idx:
            return k, self.indices[indices]
        else:
            return k

def microbatches(X, mb_size, min_size=2):
    # yields a micro-batch of objects in list X
    assert isinstance(X, 'list'), "X must be a list"
    assert len(X) > 0, f"len(X)={len(X)} == 0"
    assert all([len(x) == len(X[0]) for x in X]), "all elements must have the same length"
    N = X.size(0)
    for i in range(0, N, mb_size):
        Xb = [x[i:i+mb_size] for x in X] 
        if Xb[0].size(0) < min_size:
            continue  # skip this tiny micro-batch
        yield Xb

class BaseCalculator:
    def __init__(self, loss_module, *args, debug=False, device='cuda', **kwargs):
        self.loss_module         = loss_module
        self.debug               = debug

    def penalty(self, idxs=None, **kwargs):
        raise NotImplementedError
        
    def penalty_finalize(self, grads, szs):
        raise NotImplementedError

    def penalty_grads_finalize(self, grads, penalties, szs, **kwargs):
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
        """
        This is a hack, since we cannot calculate the true penalty:
        P = 1/E*sum_e(Loss_e - mu)**2 = 1/E*sum_e(Loss_e**2 - mu**2)
        Note that if we returned the uncentered penalty Loss_e**2, it'd be hard to 
        calculate mu_e since mu_e**2 != sum_i(Loss_{e,i})**2, but sum_i(Loss_{e,i})**2
        """

        return loss
        
    def penalty_finalize(self, risks, szs, for_grads=False, **kwargs):
        """
            risks:      risk per half, per env, unnormalized (1,num_partitions,num_envs)
            szs:        sizes of halves of environments
        """
        if not for_grads:
            # 'mu' is per-partition, NOT for ALL partitions
            mu = (risks / (szs+1e-12)).mean(dim=(0,2), keepdim=True)    # (1,num_partitions,1)
            return (risks / (szs+1e-12) - mu)**2 # normalized per env for macro-batch, (1, num_partitions, num_envs)
        else:
            return risks / (szs+1e-12)

    def penalty_grads_finalize(self, grads, penalties, szs, reduction='sum', **kwargs):
        """
        Given dLoss/dTheta, Loss per half, per env and their sizes calculate the combined gradient.
        dV/dTheta = d/dTheta(1/E*(Loss_e - 1/E*sum_j(Loss_j))^2) = 
                    2/E*sum_e((Loss_e - mu) * (grad_e - mu_grad) =
                    2/E*sum_e((Loss_e - mu) * grad_e
                    because:
                        2/E*sum_e((Loss_e - mu) * (grad_e - mu_grad) = 
                            2/E*[sum_e((Loss_e - mu)*grad_e) - sum_e((Loss_e - mu)*mu_grad)] =
                            2/E*[sum_e((Loss_e - mu)*grad_e) - sum_e(Loss_e - mu)*mu_grad]    =
                            2/E*[sum_e((Loss_e - mu)*grad_e) - 0*mu_grad] =
                            2/E*sum_e((Loss_e - mu)*grad_e) 
                    where: mu = 1/E*sum_e(Loss_e), mu_grad = 1/E*sum_e(grad_e), 
                           Loss_e = 1/N_e*sum_i(L_{e,i}), grad_e = 1/N_e*sum_i(grad_{e,i})
            grads:      dPenalty/dTheta per half, per env, unnormalized (1,num_partitions,num_envs,parnums), unweighted
            penalties:  Penalty per half, normalized per env, (1,num_partitions,num_envs), unweighted
            szs:        sizes of halves of environments (single half required)
        """
        
        num_halves, num_partitions, num_env = szs.size()
        assert num_halves == 1, "VREx number of halves should be 1"
        
        # 'mu' is per-partition, NOT for ALL partitions
        mu = penalties.mean(dim=(0,2), keepdim=True) # (1,num_partitions,1)
        x  = (2 * (penalties[..., None] - mu[..., None]) 
                * (grads / (szs[..., None]+1e-12)) 
                / num_env
             ) / num_partitions            # (parnums,)
            
        if reduction == 'sum':
            x = x.sum(dim=(0,1,2))
        elif reduction == 'none':
            x = x.squeeze(0) # remove halves dim
        
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
        
    def penalty_finalize(self, penalties, szs, for_grads=False):
        if not for_grads:
            return (penalties[0] / (szs[0]+1e-12)) * (penalties[1] / (szs[1]+1e-12))  # normalized per env for macro-batch 
        else:
            penalties_copy = penalties.clone()
            penalties_copy[0] /= (szs[0]+1e-12)
            penalties_copy[1] /= (szs[1]+1e-12)
            return penalties_copy

    def penalty_grads_finalize(self, grads, penalties, szs, debug=False, reduction='sum', sigma=None, **kwargs):
        """
        Given dPenalty/dTheta, Penalty per half, per env and their sizes calculate the combined gradient.
            grads:      dPenalty/dTheta per half, per env, unnormalized, unweighted
            penalties:  Penalty per half, normalized per env, unweighted
            szs:        sizes of halves of environments
        """
        # IRM = gs1 * gs2, where gs1 and gs2 are gradients w.r.t. scaler of mean CE of halves of sample in a batch
        # dIRM/dTheta = d(gs1 * gs2)/dTheta = dgs1/dTheta * gs2 + gs1 * dgs2/dTheta

        num_halves, num_partitions, num_env = szs.size()

        sigma = sigma if sigma else 0.
        eps = sigma * torch.randn_like(penalties)
        for i in range(num_halves):
            j = (i + num_halves + 1) % num_halves
            x = (  (grads[i] / (szs[i, ..., None]+1e-12))
                 * (penalties[j, ..., None] + eps[j, ..., None])
                 / num_env 
                )
            if reduction == 'sum':
                x = x.sum(dim=(0,1)) / num_partitions  # shape (param_numel,)
            elif reduction == 'none':
                pass
            if i == 0:
                total_grad_flat = x
            else:
                total_grad_flat += x

        if debug:
            print("total_grad_flat", total_grad_flat)
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
        # s = torch.ones(batch_size, device=device, requires_grad=True)  # one s per sample
        s = torch.tensor(1.0, requires_grad=True, device=device)
        s = s.expand(batch_size)        

        # Compute g_i in a CE-specific way
        # scaler (s) multiplies a tensor (B,logits), so need to unsqueeze dim=1
        losses = self.loss_module.compute_loss_micro(idxs=idxs, scale=s.unsqueeze(1), temperature=self.irm_temp, **kwargs)
        # losses is a scalar
        g_i = torch.autograd.grad(
            losses,
            s,
            create_graph=True,  # keep graph for next loss
        )
        # g_i is a tuple w/ entries corresponding to gradients w.r.t each parameter (here - s)
        g_i = g_i[0].squeeze(0).sum() # sum the per-sample gradients into a micro-batch gradient
        return g_i

class SimSiamIRMCalculator(IRMCalculator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def penalty(self, losses, idxs=None, **kwargs):
        device = self.loss_module._representations[0].device
        # one scalar (requires grad)
        batch_size = self.loss_module.representations(idxs=idxs)[0].size(0)
        #s = torch.ones(batch_size, device=device, requires_grad=True)  # one s per sample
        s = torch.tensor(1.0, requires_grad=True, device=device)
        s = s.expand(batch_size)        

        # Compute g_i in a CE-specific way
        # normalize=False results in use of dot product instead of cosine difference 
        losses = self.loss_module.compute_loss_micro(idxs=idxs, normalize=False, **kwargs)
        grad_outputs = torch.ones(1, losses.size(0), device=device)
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

    def pre_batch(self, batch_data, *args, **kwargs):
        pass

    def pre_micro_batch(self, batch_data, **kwargs):
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


    def loss_grads_finalize(self, grads, losses, szs, reduction='sum'):
        """
            grads:  Penalty per half, unnormalized per env, weighted
            losses: Losses per half, normalized per env, unweighted
            szs:    sizes of halves of environments
        """
        num_env = prod(szs.size())
        total_grad_flat  = (  grads  
                            / (szs[..., None]+1e-12) 
                            / num_env
                           )
        if reduction == 'sum':
            total_grad_flat = total_grad_flat.sum(dim=(0,1,2))        # shape (param_numel,)
        elif reduction == 'none':
            pass                                                      # shape (half, part, env, param_numel)
        return total_grad_flat

# ---------------------------
# MoCo+SupCon Loss Module
# ---------------------------
class MoCoSupConLossModule(LossModule):
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
        self.neg_idxs = []
        if self.debug:
            self.total_pos = 0.0
            self.total_neg = 0.0
            self.total_maxneg = 0.0
            self.count = 0

    def pre_batch(self, batch_data, index_data, partitions, device='cuda'):
        self.this_batch_size = len(batch_data)
        self.queue.get(self.this_batch_size) # advance read pointer
        
        if partitions is None or (len(partitions) == 0) or (partitions[0] is None):
            return
        
        # get the dataset indices of samples in queue
        _, indexs = self.queue.get(self.queue.queue_size - self.this_batch_size, advance=False, idx=True)
        # holds per-partition lists of per-env tensors of indices into the queue
        self.neg_idxs = [[] for _ in partitions]
        for pidx, p in enumerate(partitions):
            for env in range(p.size(-1)):
                # assign_idxs returns a tensor of indices into 'indexs' in 'env' in 'p'
                self.neg_idxs[pidx].append(utils.assign_idxs(indexs, p, env)) # append the tensor of indices to envs list

    def pre_micro_batch(self, pos, transform, indexs=None, normalize=True, dataset=None, **kwargs):
        # 'indexs' are samples indices in the dataset
        """
        Calculating SupCon w/ LogSumExp:
        Step 1 - Write the positive term:
            For a fixed anchor 'i' the positive part of SupCon is:
               1/|P(i)| \sum_{p \in P(i)} logexp(s_{ip})
               logexp(s_{ip}) = s_{ip}
               Hence, 1/|P(i)| \sum_{p \in P(i)} s_{ip}
        Step 2 - Rewrite the same quantity in the form used by the code
            Now apply a pure algebraic identity (no approximations):
                1/|P(i)| sum_{p \in P(i)} s_{ip} = A = B - [B - A] =
                log(1/|P(i)| \sum_{p \in P(i)|} exp(s_{ip})) - [log(1/|P(i)| \sum_{p \in P(i)| exp(s_ip)) - 1/|P(i)| sum_{p \in P(i)} s_{ip}]
            Now regroup:
                log(\sum_{p \in P(i)|} exp(s_{ip})) - log(|P(i)|) - C_i
            where:
                C_i = log(1/|P(i)| \sum_{p \in P(i)| exp(s_{ip})) - 1/|P(i)| \sum_{p \in P(i)} s_{ip}
            and C_i depends only on the positives of anchor i.
        Step 3 - Plug Step 2 into the full SupCon loss:
            Full SupCon loss for anchor i:
                l_i = -1/|P(i)| \sum_{p \in P(i)} s_{ip} + log(\sum_{a \in A} exp(s_ia}))
            Substitute the expression from Step 2:
                1/|P(i)| sum_{p \in P(i)} s_{ip} = log(\sum_{p \in P(i)|} exp(s_ip)) - log(|P(i)|) - C_i
            So:
                l_i = -log(\sum_{p \in P(i)|} exp(s_ip)) + log(\sum_{a \in A} exp(s_ia})) + log(|P(i)|) + C_i
        Step 4 - The code drops C_i
            The implementation uses:
                -logsum_pos + logsum_all - log(|P(i)|)
           During training the gradients of this method and standard SupCon differ.
            * The implementation is not algebraically identical to the definition
            * It is a surrogate objective
            * The surrogate has the same global minimizers as the original 
            * It does not produce the same optimization path            
        Step 5 - Why this helps TerraInc specifically
            TerraInc has:
                * strong domain shortcuts (background, color, texture)
                * same-class / different-domain positives are hard
                * many positives per anchor
            Under LSE SupCon:
                * same-domain positives stop contributing early
                * hardest (cross-domain) positives dominate gradients
                * embedding becomes domain-invariant
            This is exactly why:
                * kNN with large k improves
                * class clusters become tighter
                * domain leakage decreases
        Step 6 - In the LSE SupCon surrogate, the gradient does not explicitly depend on |P(i)|.
            As |P(i)| increases:
                True SupCon:
                    * each positive gets weaker pull
                        dl / ds_{ip} = -1/|P(i)| + softmax_A(s_{ip})
                        -> As |P(i)| grows, each positive is weakened, including the hard cross-domain ones.
            LSE SupCon:
                    * hardest positives still dominate
                    * total pull does not dilute
                    dl / ds_{ip} = -softmax_{P(i)}(s_{ip}) + softmax_A(s_{ip})                    
                    -> Hard positives dominate regardless of how many easy (same-domain) positives exist.
            LSE SupCon prevents easy same-domain positives from diluting the gradient, forcing alignment 
            to the hardest cross-domain positives - exactly what TerraInc needs to break domain clustering.
        """
        assert indexs is not None, 'indexs cannot be None'
        assert len(pos) == len(indexs), f"len(pos) {len(pos)} != len(indexs) {len(indexs)}"
        pos_q = transform(pos)
        pos_k = transform(pos)

        _, out_q = self.net(pos_q)
        if normalize:
            out_q = F.normalize(out_q, dim=1)
        with torch.no_grad():
            _, out_k = self.net_momentum(pos_k)
            if normalize:
                out_k = F.normalize(out_k, dim=1)
        
        k_queue, idx_queue = self.queue.get((self.queue.queue_size - self.this_batch_size), advance=False, idx=True)
        k_all = torch.cat([out_k, k_queue], dim=0) # (N,D), N=B+K 
        
        def get_targets(idcs, dataset, device):
            targets = [dataset.targets[i] for i in idcs]
            if dataset.target_transform is not None:
                labels = [dataset.target_transform(t) for t in targets]
            else:
                labels = targets
            return torch.tensor(labels, dtype=torch.long, device=device)
        y_batch = get_targets(indexs, dataset, pos.device)
        y_queue = get_targets(idx_queue, dataset, pos.device)
        y_all = torch.cat([y_batch, y_queue], dim=0) # (N,)

        logits = (out_q @ k_all.T) / self.temperature # (B,N)
        
        pos_mask = (y_batch[:, None] == y_all[None, :])   # (B,N)
        pos_mask[:, :len(y_batch)].fill_diagonal_(False)  # remove self-keys
        num_pos = pos_mask.sum(dim=1, keepdim=True).clamp(min=1)

        # Replace non-positives with -inf
        pos_logits = logits.masked_fill(~pos_mask, -1e9)
        # One logit per anchor = logsumexp over positives
        l_pos = torch.logsumexp(pos_logits, dim=1, keepdim=True) #- num_pos.log() # (B,1)
        
        l_neg = logits.masked_fill(pos_mask, -1e9) # (B,N)
        
        self._logits = torch.cat([l_pos, l_neg], dim=1) # (B,N+1)
        self.labels = torch.zeros(self._logits.size(0), dtype=torch.long, device=self._logits.device)
        if self.debug:
            self.total_pos    += l_pos.mean().item() * l_pos.size(0)
            self.total_neg    += l_neg.mean().item() * l_pos.size(0)
            self.total_maxneg += l_neg.max().item()  * l_pos.size(0)
            self.count        += l_pos.size(0)

        # save in state for queue update at end of batch
        self.out_k        = out_k 
        self.out_k_indexs = indexs

        self.l_pos = l_pos
        self.l_neg = l_neg

    def logits(self, idxs=None):
        if idxs is None:
            idxs = torch.arange(self._logits.size(0), device=self._logits.device)
        return self._logits[idxs]
        
    def targets(self, idxs=None):
        if idxs is None:
            idxs = torch.arange(self._logits.size(0), device=self._logits.device)
        return self.labels[idxs]

    def compute_loss_micro(self, p=None, env=None, idxs=None, scale=1.0, temperature=None, reduction='sum', **kwargs):
        # 'idxs' selects the POSITIVES in the batch
        # 'p', 'env' select the NEGATIVES in the queue
        if idxs is None:
            idxs = torch.arange(self._logits.size(0), device=self._logits.device)
        l_pos = self.l_pos
        l_neg = self.l_neg
        if p is not None:
            l_neg = l_neg[:, self.neg_idxs[p][env]]
        self._logits = torch.cat([l_pos, l_neg], dim=1) # (B,N'+1)
        self.labels = torch.zeros(self._logits.size(0), dtype=torch.long, device=self._logits.device)
        if self.debug:
            self.total_pos    += l_pos.mean().item() * l_pos.size(0)
            self.total_neg    += l_neg.mean().item() * l_pos.size(0)
            self.total_maxneg += l_neg.max().item()  * l_pos.size(0)
            self.count        += l_pos.size(0)

        # sum over batch, per env handled by driver
        # get the samples that have POSITIVES (column 0)
        l_pos = self._logits[idxs][:, 0]
        valid = l_pos > -1e9

        loss = F.cross_entropy(scale * self._logits[idxs][valid], self.labels[idxs][valid], reduction=reduction)
        return loss

    def post_micro_batch(self):
        self.queue.update(self.out_k.detach(), idx=self.out_k_indexs)

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

    @staticmethod
    def is_per_env():
        return True

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
        self.neg_idxs = []
        if self.debug:
            self.total_pos = 0.0
            self.total_neg = 0.0
            self.total_maxneg = 0.0
            self.count = 0

    def pre_batch(self, batch_data, index_data, partitions, device='cuda'):
        self.this_batch_size = len(batch_data)
        self.queue.get(self.this_batch_size) # advance read pointer
        
        if partitions is None or (len(partitions) == 0) or (partitions[0] is None):
            return
        
        # get the dataset indices of samples in queue
        _, indexs = self.queue.get(self.queue.queue_size - self.this_batch_size, advance=False, idx=True)
        # holds per-partition lists of per-env tensors of indices into the queue
        self.neg_idxs = [[] for _ in partitions]
        for pidx, p in enumerate(partitions):
            for env in range(p.size(-1)):
                # assign_idxs returns a tensor of indices into 'indexs' in 'env' in 'p'
                self.neg_idxs[pidx].append(utils.assign_idxs(indexs, p, env)) # append the tensor of indices to envs list

    def pre_micro_batch(self, pos, transform, indexs=None, normalize=True, **kwargs):
        assert indexs is not None, 'indexs cannot be None'
        assert len(pos) == len(indexs), f"len(pos) {len(pos)} != len(indexs) {len(indexs)}"
        pos_q = transform(pos)
        pos_k = transform(pos)

        _, out_q = self.net(pos_q)
        if normalize:
            out_q = F.normalize(out_q, dim=1)
        with torch.no_grad():
            _, out_k = self.net_momentum(pos_k)
            if normalize:
                out_k = F.normalize(out_k, dim=1)
        
        l_pos = torch.sum(out_q * out_k, dim=1, keepdim=True)
        l_neg = torch.matmul(out_q, self.queue.get((self.queue.queue_size - self.this_batch_size), advance=False).t())
        self._logits = torch.cat([l_pos, l_neg], dim=1)
        self.labels = torch.zeros(self._logits.size(0), dtype=torch.long, device=self._logits.device)
        if self.debug:
            self.total_pos    += l_pos.mean().item() * l_pos.size(0)
            self.total_neg    += l_neg.mean().item() * l_pos.size(0)
            self.total_maxneg += l_neg.max().item()  * l_pos.size(0)
            self.count        += l_pos.size(0)

        # save in state for queue update at end of batch
        self.out_k        = out_k 
        self.out_k_indexs = indexs

        self.l_pos = l_pos
        self.l_neg = l_neg

    def logits(self, idxs=None):
        if idxs is None:
            idxs = torch.arange(self._logits.size(0), device=self._logits.device)
        return self._logits[idxs]
        
    def targets(self, idxs=None):
        if idxs is None:
            idxs = torch.arange(self._logits.size(0), device=self._logits.device)
        return self.labels[idxs]

    def compute_loss_micro(self, p=None, env=None, idxs=None, scale=1.0, temperature=None, reduction='sum', **kwargs):
        # 'idxs' selects the POSITIVES in the batch
        # 'p', 'env' select the NEGATIVES in the queue
        if idxs is None:
            idxs = torch.arange(self._logits.size(0), device=self._logits.device)
        l_pos = self.l_pos
        l_neg = self.l_neg
        if p is not None:
            l_neg = l_neg[:, self.neg_idxs[p][env]]
        self._logits = torch.cat([l_pos, l_neg], dim=1)
        self.labels = torch.zeros(self._logits.size(0), dtype=torch.long, device=self._logits.device)
        if self.debug:
            self.total_pos    += l_pos.mean().item() * l_pos.size(0)
            self.total_neg    += l_neg.mean().item() * l_pos.size(0)
            self.total_maxneg += l_neg.max().item()  * l_pos.size(0)
            self.count        += l_pos.size(0)

        # sum over batch, per env handled by driver
        temperature = temperature or self.temperature
        loss = F.cross_entropy(scale * self._logits[idxs] / temperature, self.labels[idxs], reduction=reduction)
        return loss

    def post_micro_batch(self):
        self.queue.update(self.out_k.detach(), idx=self.out_k_indexs)

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

    @staticmethod
    def is_per_env():
        return True

# ---------------------------
# SimSiam Loss Module
# ---------------------------
class SimSiamLossModule(LossModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pre_micro_batch(self, pos, transform, normalize=True, **kwargs):
        x1 = transform(pos)
        x2 = transform(pos)

        _, z1 = self.net(x1, normalize=False)
        _, z2 = self.net(x2, normalize=False)
        p1 = self.net.module.predictor(z1, normalize=False)
        p2 = self.net.module.predictor(z2, normalize=False)
        self._representations = (z1, z2, p1, p2)

    def compute_loss_micro(self, idxs=None, scale=1.0, reduction='sum', normalize=True):
        """
        Computes unnormalized loss of a micro-batch
        """
        z1, z2, p1, p2 = self._representations
        if idxs is None:
            idxs = torch.arange(z1.size(0), device=z1.device)
        # symmetric SimSiam loss (neg cosine, average two directions)
        if normalize:
            loss_dir1 = - F.cosine_similarity(scale * p1[idxs], z2[idxs].detach(), dim=-1)
            loss_dir2 = - F.cosine_similarity(scale * p2[idxs], z1[idxs].detach(), dim=-1)
        else:
            loss_dir1 = - F.dot(scale * p1[idxs], z2[idxs].detach(), dim=-1)
            loss_dir2 = - F.dot(scale * p2[idxs], z1[idxs].detach(), dim=-1)
        loss = 0.5 * (loss_dir1 + loss_dir2)
        if reduction == 'sum':
            loss = loss.sum()
        return loss

    def representations(self, idxs=None):
        if idxs is None:
            idxs = torch.arange(self._representations[0].size(0), device=self._representations[0].device)
        return tuple(r[idxs] for r in self._representations)

    @staticmethod
    def is_per_env():
        return False

# ---------------------------
# CE Loss Module
# ---------------------------
class CELossModule(LossModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pre_micro_batch(self, pos, transform, normalize=True, labels=None, weights=None, **kwargs):
        x = transform(pos)

        features, _ = self.net(x)
        out = self.net.module.second_fc(features)
        if normalize:
            out = F.normalize(out, dim=1)
        self._logits = out
        self.labels = labels
        self.weights = weights

    def compute_loss_micro(self, normalize=True, **kwargs):
        """
        Computes unnormalized loss of a micro-batch
        """
        out = self._logits
        loss = F.cross_entropy(self._logits, self.labels, reduction='sum', weight=self.weights)
        return loss


    @staticmethod
    def is_per_env():
        return False

def clamp_scalers_for_progress(norm2_dict, dot_dict, scaler_dict, ema=False):
    def consistent_dots(dot_dict, norm2_dict, eps=1e-12):
        for (i,j) in [('k','l'), ('k','p'), ('l','p')]:
            ub = (norm2_dict[i] * norm2_dict[j]).sqrt() + eps
            dot_dict[i+j] = torch.clamp(dot_dict[i+j], -ub, ub)
            dot_dict[j+i] = dot_dict[i+j]
        return dot_dict

    scaler_dict = {k: v.clone() for k,v in scaler_dict.items()}  # make a safe copy

    if ema:
        # enforce geometric consistency
        dot_dict = consistent_dots(dot_dict, norm2_dict)
        
    LB_kl = dot_dict['kl'] / norm2_dict['k'] if norm2_dict['k'] > 0 else None
    LB_kp = dot_dict['kp'] / norm2_dict['k'] if norm2_dict['k'] > 0 else None
    LB_lp = dot_dict['lp'] / norm2_dict['l'] if norm2_dict['l'] > 0 else None
    
    UB_kl = norm2_dict['k'] / dot_dict['kl'] if dot_dict['kl'] > 0 else None
    UB_kp = norm2_dict['k'] / dot_dict['kp'] if dot_dict['kp'] > 0 else None
    UB_lp = norm2_dict['l'] / dot_dict['lp'] if dot_dict['lp'] > 0 else None
    
    q_kl  = scaler_dict['k'] / scaler_dict['l']
    q_kp  = scaler_dict['k'] / scaler_dict['p']
    q_lp  = scaler_dict['l'] / scaler_dict['p']

    q_kl_c  = torch.clamp(q_kl, LB_kl, UB_kl)
    q_kp_c  = torch.clamp(q_kp, LB_kp, UB_kp)
    q_lp_c  = torch.clamp(q_lp, LB_lp, UB_lp)
    
    if (q_kl_c == q_kl) and (q_kp_c == q_kp) and (q_lp_c == q_lp):
        return scaler_dict
    
    if q_lp_c != q_kl_c * q_kp_c:
        q_lp_c = q_kl_c * q_kp_c
    w_k = torch.ones_like(scaler_dict['k'])
    w_l = 1 / q_kl_c
    w_p = 1 / q_kp_c
    
    def normalize_weights(w1, w2, w3, T=3):
        ssum = w1 + w2 + w3
        return T*w1/ssum, T*w2/ssum, T*w3/ssum
        
    scaler_dict['k'], scaler_dict['l'], scaler_dict['p'] = normalize_weights(scaler_dict['k'], scaler_dict['l'], scaler_dict['p'])
    return  scaler_dict  

def clamp_scalers_for_progress_ema_safe(norm2, dot, scaler, eps=1e-12, do_print=False):

    def consistent_dots(dot_dict, norm2_dict, eps=1e-12):
        for (i,j) in [('k','l'), ('k','p'), ('l','p')]:
            ub = (norm2_dict[i] * norm2_dict[j]).sqrt() + eps
            dot_dict[i+j] = torch.clamp(dot_dict[i+j], -ub, ub)
            dot_dict[j+i] = dot_dict[i+j]
        return dot_dict

    scaler = {k: v.clone() for k,v in scaler.items()}  # make a safe copy
    
    # enforce geometric consistency - no need, calculated off cosines
    # dot = consistent_dots(dot, norm2)

    # compute correlation coefficients (dimensionless)
    rho_kl = dot['kl'] / (norm2['k']*norm2['l'] + eps).sqrt()
    rho_kp = dot['kp'] / (norm2['k']*norm2['p'] + eps).sqrt()
    rho_lp = dot['lp'] / (norm2['l']*norm2['p'] + eps).sqrt()

    # safe clamping bounds (weakened to survive EMA noise)
    def safe_bounds(rho):
        if rho <= 0: return torch.tensor(0.5, device=rho.device), torch.tensor(2.0, device=rho.device)      # if anti-correlated or noisy
        f = max(min(rho, 0.999), 1e-3)
        return f, 1/f

    LB_kl, UB_kl = safe_bounds(rho_kl)
    LB_kp, UB_kp = safe_bounds(rho_kp)
    LB_lp, UB_lp = safe_bounds(rho_lp)
    
    q_kl = scaler['k'] / scaler['l']
    q_kp = scaler['k'] / scaler['p']
    q_lp = scaler['l'] / scaler['p']

    q_kl_c = torch.clamp(q_kl, LB_kl, UB_kl)
    q_kp_c = torch.clamp(q_kp, LB_kp, UB_kp)
    q_lp_c = torch.clamp(q_lp, LB_lp, UB_lp)
    if do_print:
        print("q_kl", q_kl.item(), q_kl_c.item(), LB_kl.item(), UB_kl.item())
        print("q_kp", q_kp.item(), q_kp_c.item(), LB_kp.item(), UB_kp.item())
        print("q_lp", q_lp.item(), q_lp_c.item(), LB_lp.item(), UB_lp.item())

    # multiplicative consistency
    #if not torch.allclose(q_lp_c, q_kl_c*q_kp_c, atol=1e-6):
    #    q_lp_c = q_kl_c * q_kp_c

    # renormalize back to sum T=3
    w_k = torch.ones_like(scaler['k'])
    w_l = 1 / q_kl_c
    w_p = 1 / q_kp_c
    ssum = w_k + w_l + w_p
    scaler['k'] = 3 * w_k / ssum
    scaler['l'] = 3 * w_l / ssum
    scaler['p'] = 3 * w_p / ssum
    return scaler

def group_name_moco(name: str) -> str:
    """Map param name to logical block for MoCo w/ ResNet backbone and g projection head."""
    name = name.removeprefix("module.")
    if name.startswith("f.conv1") or name.startswith("f.bn1"):
        return "stem"
    if name.startswith("f.layer1"):
        return "layer1"
    if name.startswith("f.layer2"):
        return "layer2"
    if name.startswith("f.layer3"):
        return "layer3"
    if name.startswith("f.layer4"):
        return "layer4"
    if name.startswith("g."):
        return "proj_head"
    return "other"

def _ensure_grad_dict(model, grads: Union[Dict[str, torch.Tensor], List[torch.Tensor]]):
    """
    Convert grads (dict or list) into an ordered dict mapping parameter name -> grad tensor.
    If grads is a list, it must be in the same order as model.parameters().
    """
    assert isinstance(grads, dict) or isinstance(grads, list), f"Grads must be dict or list. Got {type(grads)}"
    grad_dict = {}
    if isinstance(grads, dict):
        # assume keys are param names
        grad_dict = grads
    else:
        # list-like: zip model.named_parameters() with grads list
        grad_dict = {}
        it = iter(grads)
        for (name, p) in model.named_parameters():
            try:
                g = next(it)
            except StopIteration:
                raise ValueError("grads list shorter than model.parameters()")
            grad_dict[name] = g
        # ensure no extra grads left
        try:
            next(it)
            raise ValueError("grads list longer than model.parameters()")
        except StopIteration:
            pass
    return grad_dict

def rotate_pen_toward_orthogonal(pen_grads, loss_grads, theta=0.2):
    # pen_grads, loss_grads: lists of tensors [g_env0, g_env1, ..., g_env{E-1}]
    #                        each flattened to shape (D,)
    # theta in radians (0.1-0.3 recommended; 0.2 ~ 11.5 degrees)
    # returns list of new_pen_grads (D,)
    # compute projection of loss onto pen: proj = (dot(loss,pen))/(dot(pen,pen)) * pen
    rotated = []
    device  = pen_grads[0].device
    for p,l in zip(pen_grads, loss_grads):
        denom = (p * p).sum() + 1e-12          # ()
        proj = ( (l * p).sum() / denom ) * p   # (D,) 
        loss_orth = l - proj                   # (D,)
        # normalize orth component safely
        loss_orth_norm = torch.norm(loss_orth).clamp_min(1e-12)
        loss_orth_unit = loss_orth / loss_orth_norm # (D,)
        pen_norm = torch.norm(p)
        cos_t = torch.cos(torch.tensor(theta, device=device))
        sin_t = torch.sin(torch.tensor(theta, device=device))
        new_pen = cos_t * p + sin_t * (pen_norm * loss_orth_unit) # (D,)
        rotated.append(new_pen)
    return rotated

def calculate_loss_grads_final(loss_grads, loss_env, loss_weight_env, halves_sz, loss_module, reduction, device, do_loss):
    if do_loss:
        loss_grads_final = []
        for pind in range(len(loss_grads)):
            dLoss_dTheta_env = loss_grads[pind] * loss_weight_env[..., None]  # per env sum of dCont/dTheta, shape (I,J,K,param_numel), unweighted
            reduction = 'sum'
            total_grad_flat  = loss_module.loss_grads_finalize(dLoss_dTheta_env, loss_env, halves_sz, reduction=reduction)
            loss_grads_final.append(total_grad_flat)
    else:
        loss_grads_final = [torch.tensor(0., dtype=torch.float, device=device)] * len(loss_grads)
    return loss_grads_final

def calculate_penalty_grads_final(penalty_grads, penalty_aggregator, penalty_weight_env, halves_sz, penalty_calculator, reduction, device, do_penalty):
    if do_penalty:
        penalty_grads_final = []
        pen = penalty_calculator.penalty_finalize(penalty_aggregator, halves_sz, for_grads=True) # normalized per env for macro-batch, unweighted
        for pind in range(len(penalty_grads)):
            dPenalty_dTheta_env = penalty_grads[pind] * penalty_weight_env[..., None] # per env sum of dPenalty/dTheta over macro-batch per parameter, unweighted, shape (I,J,K,param_numel)
            reduction = 'sum'
            total_grad_flat     = \
                penalty_calculator.penalty_grads_finalize(
                    dPenalty_dTheta_env, 
                    pen, 
                    halves_sz,
                    sigma=args.penalty_sigma,
                    reduction=reduction
                )                                                                     
            penalty_grads_final.append(total_grad_flat.detach())
    else:
        penalty_grads_final = [torch.tensor(0., dtype=torch.float, device=device)] * len(penalty_grads)
    return penalty_grads_final

def rotate_penalty_grads(penalty_grads_final, loss_grads_final, grad_rotate, do_penalty):
    if do_penalty and grad_rotate is not None:
        theta = float(torch.empty(1).uniform_(min(grad_rotate), max(grad_rotate)).item())
        penalty_grads_final = [rotate_pen_toward_orthogonal( # returns list w/ 1 element, w/ size (parnum,)
            [g], [loss_grads_final[pind]], theta=theta)[0].clone() for pind, g in enumerate(penalty_grads_final)
        ]
    return penalty_grads_final

def calculate_scalers(loss_unsplit_grads_final, loss_grads_final, penalty_grads_final, 
                      loss_unsplit_aggregator,  loss_env,         penalty_env,
                      loss_unsplit_weight,      loss_weight,      penalty_weight,
                      ema,
                      gradnorm_balancer, do_gradnorm,
                      args, do_unsplit_loss, do_loss, do_penalty, device):
    def setup_grads_and_norms(grads_final, weight, Lscaler, device, do_flag, default_grads_weighted_vector=None):
        if do_flag:
            grads_weighted = [g.detach().clone() * weight * Lscaler for g in grads_final if g is not None]
            grads_weighted_vector = torch.cat([g for g in grads_weighted]) 
            grads_norm_weighted = grads_weighted_vector.norm()
        else:
            grads_weighted = [torch.zeros_like(g) for g in grads_final if g is not None]
            assert default_grads_weighted_vector is not None, "default_grads_weighted_vector not given when do_flag is False"
            grads_weighted_vector = default_grads_weighted_vector
            grads_norm_weighted = torch.tensor(0., dtype=torch.float, device=device)
        return grads_weighted, grads_weighted_vector, grads_norm_weighted

    loss_unsplit_grads_final_weighted, l_unsplit_grads_flat_weighted, loss_unsplit_grad_norm_weighted = \
        setup_grads_and_norms(loss_unsplit_grads_final, loss_unsplit_weight, args.Lscaler, device, True)
    default_grads_weighted_vector = torch.zeros_like(l_unsplit_grads_flat_weighted)
    loss_grads_final_weighted, l_grads_flat_weighted, loss_grad_norm_weighted = \
        setup_grads_and_norms(loss_grads_final, loss_weight, args.Lscaler, device, do_loss, default_grads_weighted_vector=default_grads_weighted_vector)
    penalty_grads_final_weighted, p_grads_flat_weighted, penalty_grad_norm_weighted = \
        setup_grads_and_norms(penalty_grads_final, penalty_weight, args.Lscaler, device, do_penalty, default_grads_weighted_vector=default_grads_weighted_vector)

    # Compute dot products & cosines
    delta_lk = l_grads_flat_weighted.dot(l_unsplit_grads_flat_weighted)       
    delta_lp = l_grads_flat_weighted.dot(p_grads_flat_weighted)
    delta_kp = l_unsplit_grads_flat_weighted.dot(p_grads_flat_weighted)
    cos_lk   = delta_lk / (loss_unsplit_grad_norm_weighted * loss_grad_norm_weighted    + 1e-12)
    cos_lp   = delta_lp / (loss_grad_norm_weighted      * penalty_grad_norm_weighted + 1e-12)
    cos_kp   = delta_kp / (loss_unsplit_grad_norm_weighted * penalty_grad_norm_weighted + 1e-12)

    Loss_grads_flat_weighted = [loss_unsplit_grads_final_weighted[p] + loss_grads_final_weighted[p] for p in range(len(loss_grads_final_weighted))]
    L_grads_flat_weighted = l_unsplit_grads_flat_weighted + l_grads_flat_weighted
    cos_Lp   = F.cosine_similarity(L_grads_flat_weighted, p_grads_flat_weighted, dim=0)
    dot_Lp   = L_grads_flat_weighted.dot(p_grads_flat_weighted)

    loss_weighted      = loss_weight      * loss_env.mean()
    loss_unsplit_weighted = loss_unsplit_weight * loss_unsplit_aggregator.mean()
    penalty_weighted   = penalty_weight   * penalty_env.mean()

    if args.ema:
        emas = ema.update({'ngk':      loss_unsplit_grad_norm_weighted, 
                           'ngl':      loss_grad_norm_weighted, 
                           'ngp':      penalty_grad_norm_weighted, 
                           'cos_lk':   cos_lk,
                           'cos_lp':   cos_lp,
                           'cos_kp':   cos_kp
                          }, orig_shape=True)   # return data shaped as input data
        # make sure the order is explicit and not some implicit one
        emas_k = ['ngk', 'ngl', 'ngp', 'cos_lk', 'cos_lp', 'cos_kp']
        ngk, ngl, ngp, cos_lk, cos_lp, cos_kp = [emas[k] for k in emas_k]
        dot_lk = ngk * ngl * cos_lk
        dot_lp = ngl * ngp * cos_lp
        dot_kp = ngk * ngp * cos_kp
    else:
        ngk      = loss_unsplit_grad_norm_weighted
        ngl      = loss_grad_norm_weighted
        ngp      = penalty_grad_norm_weighted
        dot_lk   = delta_lk
        dot_lp   = delta_lp
        dot_kp   = delta_kp

    # This awlays holds because we compute it from cosines
    assert dot_lk.abs() <= ngk * ngl, f"ngk {ngk}, ngl {ngl}, lk {dot_lk.abs()}" 
    assert dot_lp.abs() <= ngl * ngp, f"ngl {ngl}, ngp {ngp}, lp {dot_lp.abs()}"
    assert dot_kp.abs() <= ngk * ngp, f"ngk {ngk}, ngp {ngp}, lpk {dot_lp.abs()}"

    ngk2 = ngk ** 2
    ngl2 = ngl ** 2
    ngp2 = ngp ** 2

    normalized_scales = {}
    gradnorm_rates = torch.zeros(int(args.penalty_weight>0) + int(do_loss) + int(do_unsplit_loss), dtype=torch.float, device=device)
    losses_dict, grad_norms_dict = {}, {}
    if do_penalty:
        losses_dict['penalty']       = penalty_weighted
        grad_norms_dict['penalty']   = ngp
    if do_loss:
        losses_dict['loss']          = loss_weighted
        grad_norms_dict['loss']      = ngl
    if do_unsplit_loss:
        losses_dict['loss_unsplit']     = loss_unsplit_weighted
        grad_norms_dict['loss_unsplit'] = ngk            

    if do_gradnorm:
        normalized_scales, gradnorm_loss, gradnorm_rates, gradnorm_progress = gradnorm_balancer.compute_weights_and_loss(losses_dict, grad_norms_dict)
        """
        print()
        print([f'{k}: {v.item()}' for k,v in normalized_scales.items()], 
               f'gloss: {gradnorm_loss.item()}', 
               [f'{k}: {gradnorm_rates[i].item()}' for i,k in enumerate(task_names)], 
               [f'{k} {gradnorm_balancer.task_weights[k].item()}' for k in task_names])
        """
    else:
        normalized_scales = {k: torch.tensor(v, dtype=torch.float, device=device) for k,v in args.gradnorm_scalers.items()}
        gradnorm_progress = torch.tensor(1.0, dtype=torch.float)
        gradnorm_loss = torch.tensor(1.0, dtype=torch.float)

    task_names   = gradnorm_balancer.task_names # list
    task_names_2_klp = {'loss_unsplit': 'k', 'loss': 'l', 'penalty': 'p'}
    if args.clamp_weights_for_progress:
        dot_dict    = {'kl': dot_lk, 'kp': dot_kp, 'lp': dot_lp}
        norm2_dict  = {'k':  ngk2,   'l':  ngl2,   'p':  ngp2}
        scaler_dict = {v: normalized_scales[k] for k,v in task_names_2_klp.items()}
        #w = clamp_scalers_for_progress(norm2_dict, dot_dict, scaler_dict, ema=(args.ema is not None))
        w = clamp_scalers_for_progress_ema_safe(norm2_dict, dot_dict, scaler_dict, do_print=args.debug)
        # this can CHANGE the relative rank of the weights!!!
        normalized_scales = {k: w[v] for k,v in task_names_2_klp.items()} 

    loss_unsplit_grad_scaler = normalized_scales['loss_unsplit'] if do_unsplit_loss else torch.tensor(1.0, dtype=torch.float, device=device)
    loss_grad_scaler      = normalized_scales['loss']      if do_loss      else torch.tensor(1.0, dtype=torch.float, device=device)
    penalty_grad_scaler   = normalized_scales['penalty']   if do_penalty   else torch.tensor(1.0, dtype=torch.float, device=device)

    gn_pm = 0
    for pind, p in enumerate(gradnorm_balancer.parameters()):
        if p.grad is not None:
            gn_pm += (2**pind)*(p.grad.sign()) 

    gradnorm_rates  = gradnorm_rates.tolist() if do_gradnorm else []
    info_dict = {
        'ngk':               ngk.item(),
        'ngl':               ngl.item(),
        'ngp':               ngp.item(),
        'dot_lk':            dot_lk.item(),               
        'dot_lp':            dot_lp.item(),
        'dot_kp':            dot_kp.item(),
        'cos_lk':            cos_lk.item(),               
        'cos_lp':            cos_lp.item(),
        'cos_kp':            cos_kp.item(),
        'gradnorm_loss':     gradnorm_loss.item()    if do_gradnorm else 0.,
        'ngk2':              ngk2.item(),
        'ngl2':              ngl2.item(),
        'ngp2':              ngp2.item(),
        'cos_Lp':            cos_Lp.item(),
        'dot_Lp':            dot_Lp.item(),
        'gradnorm_progress': gradnorm_progress.item(),
        # get these before gradnorm optimizer updates them
        'w_k':               loss_unsplit_grad_scaler.item(),
        'w_l':               loss_grad_scaler.item(),
        'w_p':               penalty_grad_scaler.item(),
        'v_k':               gradnorm_balancer.task_weights['loss_unsplit'].item() if 'loss_unsplit' in gradnorm_balancer.task_weights else 0.,
        'v_l':               gradnorm_balancer.task_weights['loss'].item()      if 'loss'      in gradnorm_balancer.task_weights else 0.,
        'v_p':               gradnorm_balancer.task_weights['penalty'].item()   if 'penalty'   in gradnorm_balancer.task_weights else 0.,
        'gn_pm':             gn_pm,
        # if cond > 0, the corresponding quantity would decrease
        'loss_decrease_cond':      loss_grad_scaler      * ngl2 + loss_unsplit_grad_scaler*dot_lk + penalty_grad_scaler*dot_lp,
        'loss_unsplit_decrease_cond': loss_unsplit_grad_scaler * ngk2 + loss_grad_scaler*dot_lk      + penalty_grad_scaler*dot_kp,
        'penalty_decrease_cond':   penalty_grad_scaler   * ngl2 + loss_unsplit_grad_scaler*dot_kp + loss_grad_scaler*dot_lp,
        'gradnorm_rates_str':     " ".join([f'{n} {r:.4f}' for n,r in zip([task_names_2_klp[k] for k in task_names], gradnorm_rates)]) if do_gradnorm else "",  
    }

    return loss_unsplit_grad_scaler, loss_grad_scaler, penalty_grad_scaler, gradnorm_loss, info_dict

def gradnorm_update(gradnorm_balancer, gradnorm_loss, gradnorm_optimizer, args, do_gradnorm):
    if not do_gradnorm:
        return
    gradnorm_optimizer.zero_grad(set_to_none=True) # clear gradients
    gradnorm_loss.backward()
    gradnorm_balancer.remove_common_mode_hook()    # remove common-mode from grads

    # actual computed grads after backward:
    if args.gradnorm_debug and 'gn' in args.gradnorm_debug:
        with np.printoptions(precision=6):
            print("actual v.grad:\t", np.array([gradnorm_balancer.task_weights[k].grad.item() for k in gradnorm_balancer.task_names]))

    gradnorm_optimizer.step()

    if args.gradnorm_debug and 'opt' in args.gradnorm_debug:
        print()
        opt_ids = {id(p) for g in gradnorm_optimizer.param_groups for p in g['params']}
        # 1) Does optimizer actually contain the exact Parameter objects?
        for k, p in gradnorm_balancer.task_weights.items():
            print("param in opt?", k, id(p) in opt_ids)

        # 2) Are grads present and nonzero?
        for k, p in gradnorm_balancer.task_weights.items():
            print(k, "requires_grad=", p.requires_grad,
                  "grad is None?", p.grad is None,
                  "grad norm=", None if p.grad is None else p.grad.norm().item())

        # 3) Check loss and dtype/device sanity
        print("loss item:", float(gradnorm_loss.item()))
        print("weights device/dtype:", [(k, p.device, p.dtype) for k, p in gradnorm_balancer.task_weights.items()])

        # 4) Check lr / optimizer param count
        print("optimizer lr(s):", [g['lr'] for g in gradnorm_optimizer.param_groups])
        print("optimizer param counts:", [len(g['params']) for g in gradnorm_optimizer.param_groups])

    lb = {'loss_unsplit': 1e-3,  'loss': 1e-3,  'penalty': 1e-3} 
    ub = {'loss_unsplit': 5.0,   'loss': 5.0,   'penalty': 5.0} 
    gradnorm_balancer.clamp_weights(lb, ub) # clamps UNNORMALIZED weights
    if (all_lb := all([v == lb[k] for k,v in gradnorm_balancer.task_weights.items()])) or \
       (all_ub := all([v == ub[k] for k,v in gradnorm_balancer.task_weights.items()])):
       bound = "LB" if all_lb else "UB"
       warnings.warn(f"[GN WARNING] All unnormalized weights clamped to {bound}. Resetting.")
       gradnorm_balancer.rescale_weights()
       utils.reset_optimizer(gradnorm_optimizer)

def make_rand_dither_weight(num_partitions, env_num, weight_env_eps, device):
    deltas = (2.0 * torch.rand(1, num_partitions, env_num, device=device) - 1.0) * weight_env_eps  # random in [-eps, eps]
    deltas -= deltas.mean()  # mean-zero so overall scale unchanged
    weight_env = 1.0 + deltas
    return weight_env
    
def set_BN_adapt(net, adapt_bn, bn_momentum)
    for m in net.modules():
        if isinstance(m, (torch.nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
            if not adapt_bn:
                m.eval()                 # use stored running stats
                m.track_running_stats = True
            else:
                m.train()                 # learn pars
                m.track_running_stats = True
                m.momentum = bn_momentum

# ssl training with IP-IRM
def train_env(net, train_loader, train_optimizer, partitions, batch_size, args, **kwargs):

    net.train()
    set_BN_adapt(net, args.adapt_bn, args.bn_momentum)

    if isinstance(partitions, list): # if retain previous partitions
        assert args.retain_group
    else:
        partitions = [partitions]
    num_partitions = len(partitions)
    
    device = next(net.parameters()).device

    transform        = train_loader.dataset.transform
    target_transform = train_loader.dataset.target_transform

    if args.increasing_weight:
        penalty_weight = utils.increasing_weight(args.increasing_weight, args.penalty_weight, args.penalty_iters, epoch, args.epochs)
    elif args.penalty_iters < 200:
        penalty_weight = args.penalty_weight if epoch >= args.penalty_iters else 0.
    else:
        penalty_weight = args.penalty_weight
        
    loss_weight         = args.penalty_cont             * (1 if penalty_weight <= 1 else 1 / penalty_weight)
    loss_unsplit_weight = max(args.penalty_unsplit_cont * (1 if penalty_weight <= 1 else (1 / penalty_weight)), int(args.baseline))
    penalty_weight_orig = penalty_weight
    penalty_weight      = 1 if penalty_weight > 1 else penalty_weight
    
    do_loss         = (not args.baseline) and (loss_weight > 0)
    do_unsplit_loss = (args.baseline)     or ((args.unsplit_cont)  and (loss_unsplit_weight > 0))
    do_penalty      = (not args.baseline) and (penalty_weight > 0)
    do_gradnorm     = (not args.baseline) and args.gradnorm        and (epoch >= args.gradnorm_epoch)

    loader_batch_size            = batch_size
    gradients_accumulation_steps = args.gradients_accumulation_batch_size // loader_batch_size 
    gpu_batch_size               = args.micro_batch_size
    gpu_accum_steps              = ceil(loader_batch_size / gpu_batch_size) # better round up 

    gradients_accumulation_step  = 0
    alternating_gradients_update = 0
    total_samples                = len(train_loader.dataset)
    
    trained_samples             = 0
    total_unsplit_loss_weighted = 0.0
    total_env_loss_weighted     = 0.0
    total_irm_loss_weighted     = 0.0
    total_loss_weighted         = 0.0

    bar_format = '{l_bar}{bar:' + str(args.bar) + '}{r_bar}' #{bar:-' + str(args.bar) + 'b}'
    train_bar = tqdm(train_loader,
            total=len(train_loader),        # number of batches
            ncols=args.ncols,               # total width available
            dynamic_ncols=False,            # disable autosizing
            bar_format=bar_format,          # request bar width
            )

    # instantiate LossModule and IRMCalculator based on args (pluggable)
    # default to MoCo if args.loss_type not provided
    loss_type = getattr(args, 'ssl_type', 'moco').lower()
    penalty_type = getattr(args, 'penalty_type', 'irm').lower()
    loss_unsplit_type = getattr(args, 'loss_unsplit_type', None)
    loss_unsplit_type = loss_unsplit_type.lower() if loss_unsplit_type is not None else None

    if loss_unsplit_type == 'ce' or loss_unsplit_type == 'ceweighted':
        LossUnsplitModule = CELossModule
    elif loss_unsplit_type is None:
        LossUnsplitModule = None
    else:
        raise ValueError(f"Unknown loss_unsplit_type: {loss_unsplit_type}")

    if loss_type == 'moco':
        LossModule = MoCoLossModule
    elif loss_type == 'mocosupcon':
        LossModule = MoCoSupConLossModule
    elif loss_type == 'simsiam':
        LossModule = SimSiamLossModule
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if LossUnsplitModule is not None:
        loss_unsplit_module = LossUnsplitModule(net, debug=args.debug, **kwargs) 
    else:
        loss_unsplit_module =  None

    is_per_env  = LossModule.is_per_env()
    loss_module = LossModule(net, debug=args.debug, **kwargs) 

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

    penalty_calculator   = PenaltyCalculator(loss_module, irm_temp=args.irm_temp, debug=args.debug, **kwargs)
    """
    We made an attempt to get rid of halving the micro-batches of the whole batch into two subsets. 
    Turns out this cannot be done because losses and gradients are aggregated over micro-batches, 
    but aggregations over halves are needed for IRM and it's impossible to recover back the halves 
    from full aggregators.
    """
    num_halves  = PenaltyCalculator.num_halves()

    loss_aggregator         = torch.zeros((num_halves, num_partitions, args.env_num), dtype=torch.float, device=device) 
    penalty_aggregator      = torch.zeros((num_halves, num_partitions, args.env_num), dtype=torch.float, device=device) 
    loss_unsplit_aggregator = torch.tensor(0, dtype=torch.float, device=device) # scalar
    halves_sz               = torch.zeros((num_halves, num_partitions, args.env_num), dtype=torch.float, device=device) 
    # One buffer per parameter
    loss_grads = [  # dLoss / dTheta
        torch.zeros((*loss_aggregator.shape, p.numel()), dtype=p.dtype, device=p.device)
        for p in net.parameters()
    ]
    penalty_grads = [ # dPenalty / dTheta
        torch.zeros((*penalty_aggregator.shape, p.numel()), dtype=p.dtype, device=p.device)
        for p in net.parameters()
    ]
    # loss unsplit doesn't require finalization
    loss_unsplit_grads_final = [  # dLoss / dTheta
        torch.zeros(p.numel(), dtype=p.dtype, device=p.device)
        for p in net.parameters()
    ]

    train_optimizer.zero_grad(set_to_none=True) # clear gradients at the beginning 
    k = 0 # number of consecutive batches r_mag is within bounds
    macro_batch_index = 0

    for batch_index, data_env in enumerate(train_bar):

        reduction = 'sum' if is_per_env else 'none' # make sure it's the correct one

        data_batch, labels_batch, indexs_batch = data_env # 'data_batch' is an batch of images, 'indexs_batch' is their corresponding indices 
        this_batch_size = len(indexs_batch) # for the case drop_last=False
        
        loss_module.pre_batch(data_batch, indexs_batch, partitions)
        if loss_unsplit_module is not None:
            loss_unsplit_module.pre_batch(data_batch) # weights handled below

        # -----------------------
        # Step 0: micro-batches
        # -----------------------
        mb_list = list(microbatches([data_batch, labels_batch, indexs_batch], gpu_batch_size))

        for j in range(num_halves): # over halves of micro-batches
            for i in [i_ for i_ in range(len(mb_list)) if i_ % num_halves == j]: # loop over micro-batches
                # per micro-batch pipeline
                batch_micro, labels, indexs = mb_list[i]
                if (loss_unsplit_module is not None) and (kwargs['CEweights'] is not None):
                    weights         = kwargs['CEweights'][indexs]
                else:
                    weights         = None
                batch_micro         = batch_micro.cuda(non_blocking=True)
                labels              = labels.cuda(non_blocking=True)
                indexs              = indexs.cuda(non_blocking=True)
                weights             = weights.cuda(non_blocking=True) if weights is not None else None

                num_samples           = 1 if is_per_env else len(batch_micro)
                num_split_repeates    = int(not args.baseline) * (int(loss_weight>0) + int(penalty_weight>0)) # 0, 1 or 2
                num_baseline_repeates = int(loss_unsplit_weight>0) * int(args.unsplit_cont) # 0 or 1                                  
                num_repeats           = max(num_split_repeates, num_baseline_repeates) # max is for the case 'num_split_repeates' is 0
                num_grads             = num_partitions * args.env_num * num_split_repeates + num_baseline_repeates # number of rows
                if is_per_env:
                    # For per-env loss_cont, each env gets its own column + one more column for loss_unsplit, if requested, whether same to loss_cont or not
                    grad_outputs          = torch.zeros((num_grads, num_grads), dtype=torch.float, device=device) 
                else:
                    # for per-sample loss_cont, the number of columns is the number of samples times the number of repeats. 
                    # loss_cont and loss_unsplit share the SAME columns, if loss_unsplit is similar to loss_cont; otherwise loss_unsplit has its own extra column
                    if num_baseline_repeates and loss_unsplit_module is None:
                        grad_outputs      = torch.zeros((num_grads, num_samples*num_repeats), dtype=torch.float, device=device) 
                    elif num_baseline_repeates and loss_unsplit_module is not None:
                        grad_outputs      = torch.zeros((num_grads, num_samples*num_split_repeates + 1), dtype=torch.float, device=device) 
                    else:
                        grad_outputs      = torch.zeros((num_grads, num_samples*num_repeats), dtype=torch.float, device=device) 
                differentiate_this    = []

                """
                prepare for micro-batch in loss-sepcific way:
                    MoCo:    generate two views, get their embeddings from respective encoders, normalize them, etc
                    SimSiam: generate two views, get their projections and predictions, etc
                """
                if do_unsplit_loss and loss_unsplit_module is not None:
                    loss_unsplit_module.pre_micro_batch(batch_micro, transform=transform, indexs=indexs, labels=labels, normalize=False, 
                        dataset=train_loader.dataset, weights=weights)
                    losses_samples_all = loss_unsplit_module.compute_loss_micro(reduction='sum')
                    # Must be first to be in 1st column 
                    differentiate_this.append(losses_samples_all)

                loss_module.pre_micro_batch(batch_micro, transform=transform, indexs=indexs, normalize=(loss_type != 'supcon'), dataset=train_loader.dataset)
                
                # Even if 'do_loss'==False, when SAME loss is used for BOTH loss_cont and loss_unsplit, 'reduction' reflects the correct reduction
                if (do_unsplit_loss and loss_unsplit_module is None) or (do_loss and not is_per_env):
                    # compute unnormalized WHOLE micro-batch loss, no split into envs
                    losses_samples_all = loss_module.compute_loss_micro(reduction=reduction)
                    differentiate_this.append(losses_samples_all)

                if do_penalty and not is_per_env:
                    penalties_samples = penalty_calculator.penalty(losses_samples, reduction=reduction)
                    differentiate_this.append(penalties_samples)

                if do_loss or do_penalty:
                    for partition_num, partition in enumerate(partitions):
                        for env in range(args.env_num):

                            # split mb: 'idxs' are indices into 'indexs' that correspond to domain 'env' in 'partition'
                            # 'indexs' are the indices of samples in dataset which are in this micro-batch
                            idxs = utils.assign_idxs(indexs, partition, env)

                            if (N := len(idxs)) == 0:
                                if is_per_env:
                                    if do_loss:
                                        differentiate_this.append(torch.zeros(1, dtype=torch.float, device=device)) # dummy loss
                                    if do_penalty:
                                        differentiate_this.append(torch.zeros(1, dtype=torch.float, device=device)) # dummy penalty
                                continue
                            
                            halves_sz[j,partition_num,env] += N # update number of elements in environment
                            
                            # losses - losses are ALWAYS a scalar
                            if do_loss:
                                # compute unnormalized micro-batch loss
                                if is_per_env:
                                    # compute unnormalized micro-batch loss
                                    # 'idxs' select the samples from this micro-batch
                                    # 'loss_samples' are either the per-sample losses or thie sum depending on 'reduction'
                                    # for 'is_per_env'==True, it's always 'sum'
                                    losses_samples = loss_module.compute_loss_micro(p=partition_num, env=env, reduction=reduction, idxs=idxs)
                                    differentiate_this.append(losses_samples)
                                    loss = losses_samples.detach()
                                else:
                                    # For 'is_per_env'==False, convert per-sample losses to a sum
                                    loss = losses_samples[idxs].sum(dim=0).detach()
                                loss_aggregator[j,partition_num,env] += loss # unnormalized, before penalty scaler
                            # penalties - penalties are ALWAYS a scalar
                            if do_penalty:
                                if is_per_env:
                                    penalties_samples = penalty_calculator.penalty(losses_samples, reduction=reduction)
                                    differentiate_this.append(penalties_samples)
                                    penalty = penalties_samples.detach()
                                else:
                                    penalty = penalties_samples[idxs].sum(dim=0).detach()
                                penalty_aggregator[j,partition_num,env] += penalty # unnormalized penalty components before penalty scaler

                            # gradients
                            """
                            'grad_outputs' is a table w/ each row corresponding to a loss/penalty of an env in a partition.
                            The top half coresponds to cont losses; the bottom half corresponds to penalties.
                            The last row corresponds to loss_unsplit.
                            For per_env losses (e.g., MoCo):
                                each column corresponds to an env in a partition loss/penalty. 
                                The 1st column corresponds to loss_unsplit if requested.
                            For non-per-env losses (e.g., SimSiam):
                                each column corresponds to a sample loss/penalty.
                                The left half corresponds to losses; the right half corresponds to penalties.
                                The 1st to 'num_samples' columns correspond to loss_unsplit if requested.                                
                            'grad_outputs[i,j]' is a multiplier of the [i,j]-th entry to differentiate.
                            """

                            # 1. Calculate base indices once
                            # Using standard Python ints for indexing is faster than creating 0-d Tensors
                            base_idx = partition_num * args.env_num + env
                            row_stride = num_partitions * args.env_num

                            # 2. Determine Mask and Initial Offset
                            if is_per_env:
                                # Per-env: Mask is solid 1.0; Offset is based on the partition index
                                mask = 1.0
                                current_offset = base_idx + int(do_unsplit_loss)
                            else:
                                # Shared: Mask is sparse (zeros with ones at idxs); Offset starts at 0
                                mask = torch.zeros(num_samples, dtype=torch.float, device=device)
                                mask[idxs] = 1.0
                                current_offset = 0

                            # 3. Apply Updates Sequentially
                            current_row = base_idx

                            if do_loss:
                                # Use comma indexing [row, col] for efficiency
                                grad_outputs[current_row, current_offset : current_offset + num_samples] = mask

                                # Shift indices for the penalty step
                                current_row += row_stride
                                current_offset += num_samples

                            if do_penalty:
                                grad_outputs[current_row, current_offset : current_offset + num_samples] = mask

                        # end for env in range(args.env_num):
                    # end for partition_num, partition in enumerate(partitions):
                # end if not args.baseline:

                if do_unsplit_loss: # global loss @ 1st partition
                    # This could be done w/o the split into two halves, but this streamlines the code w/o any harm
                    # Here we know that losses are over the whole macro-batch, so we can normalize up-front
                    # sum() is for the case when 'is_per_env'==False; otherwise - it's harmless
                    # 'losses_samples_all' are the losses of ALL samples in this micro-batch
                    # loss - loss is ALWAYS a scalar
                    loss = losses_samples_all.sum().detach()  / this_batch_size / gradients_accumulation_steps
                    # compute unnormalized gradients for this loss
                    # grad_outputs: one per sample
                    loss_unsplit_aggregator += loss # before scaler

                    offset = 0 # 1st column
                    number_of_columns = num_samples if loss_unsplit_module is None else 1
                    grad_outputs[-1][offset:offset+number_of_columns]  = 1.0 / this_batch_size / gradients_accumulation_steps # unweighted

                differentiate_this = [t.reshape(-1) for t in differentiate_this] # ensure common shape of 1D tensors
                differentiate_this = torch.cat(differentiate_this, dim=0) # cat losses and penalties into a single vector length 2B

                # compute all needed grads
                # 'grads_all' is a tuple w/ an entry per parameter.
                # each entry is a tensor w/ 1st dim = 'grad_outputs.size(0)' and other dims matching the parameter

                """
                print()
                print(f"num_samples {num_samples}, num_split_repeates {num_split_repeates}, num_baseline_repeates {num_baseline_repeates}, " +                                  
                      f"num_repeats {num_repeats}, num_grads {num_grads}, number_of_columns {number_of_columns}, " + 
                      f"grad_outputs {grad_outputs.size()}, differentiate_this {differentiate_this.size()}")
                """

                # autograd sums all gradients in each row for each parameter
                grads_all = torch.autograd.grad(
                    differentiate_this,
                    tuple(net.parameters()),
                    retain_graph=False,  # no need to keep graph for next loss
                    allow_unused=True,
                    grad_outputs=grad_outputs, 
                    is_grads_batched=True
                )

                # 1. Pre-calculate bounds and offsets
                num_tasks      = (num_grads - num_baseline_repeates) // max(1, num_split_repeates)
                penalty_offset = num_partitions * args.env_num

                # 2. Single pass over parameters
                # 'grads_all' is a tuple w/ an entry per parameter.
                # each entry is a tensor w/ 1st dim = 'grad_outputs.size(0)' and other dims matching the parameter
                for _j, g in enumerate(grads_all):
                    if g is None:
                        continue

                    # Flatten: (Batch, *Param_Dims) -> (Batch, Flattened_Dim)
                    # Using reshape/view here avoids repeating it inside inner loops
                    g_flat = g.detach().reshape(g.size(0), -1)

                    # --- Keep Loss (Last Row) ---
                    if do_unsplit_loss:
                        loss_unsplit_grads_final[_j] += g_flat[-1]

                    # --- Loss & Penalty (Batch Slicing) ---
                    if (do_loss or do_penalty) and num_tasks > 0:
                        # Shape to broadcast: (Partitions, Envs, Flattened_Dim)
                        view_shape = (num_partitions, args.env_num, -1)

                        if do_loss:
                            # Slice first 'num_tasks' rows -> Reshape -> Add
                            loss_grads[_j][j] += g_flat[:num_tasks].view(view_shape)

                        if do_penalty:
                            # Slice rows starting at offset -> Reshape -> Add
                            penalty_grads[_j][j] += g_flat[penalty_offset : penalty_offset + num_tasks].view(view_shape)

                # end if not args.baseline:
                loss_module.post_micro_batch()
                loss_module.prepare_for_free()
                if loss_unsplit_module is not None:
                    loss_unsplit_module.post_micro_batch()
                    loss_unsplit_module.prepare_for_free()
                
                # free memory of micro-batch
                del batch_micro, indexs, g_flat, g, grads_all, differentiate_this
                if do_loss or do_unsplit_loss:
                    del loss
                if do_unsplit_loss:
                    del losses_samples_all
                if do_loss:
                    del losses_samples
                if do_penalty:
                    del penalties_samples, penalty
            # end for i in [i_ for i_ in range(len(mb_list)) if i_ % 2 == j]:
            torch.cuda.empty_cache()
        # end for j in range(idxs):
        torch.cuda.empty_cache()
        trained_samples += this_batch_size # total number of samples processed so far
        
        gradients_accumulation_step += 1
        if gradients_accumulation_step < gradients_accumulation_steps:
            continue
        
        macro_batch_index           += 1

        loss_weight_env    = make_rand_dither_weight(num_partitions, args.env_num, args.weight_env_eps, device)
        penalty_weight_env = make_rand_dither_weight(num_partitions, args.env_num, args.weight_env_eps, device)

        if do_loss:
            partition_sz = halves_sz.sum(dim=0, keepdim=True) # (1,J,K) # sizes of envs in macro-batch
            loss_env = loss_aggregator.sum(dim=0, keepdim=True) / partition_sz  # per env for macro-batch, normalized per env, unweighted
        else:
            loss_env = torch.tensor(0, dtype=torch.float, device=device)
        if do_penalty:
            penalty_env = penalty_calculator.penalty_finalize(penalty_aggregator, halves_sz) # normalized per env for macro-batch, unweighted
        else:
            penalty_env = torch.tensor(0, dtype=torch.float, device=device)

        # Environments gradients
        loss_grads_final = calculate_loss_grads_final(loss_grads, loss_env, loss_weight_env, halves_sz, loss_module, reduction, device, do_loss)

        penalty_grads_final = calculate_penalty_grads_final(penalty_grads, penalty_aggregator, penalty_weight_env, halves_sz, penalty_calculator, reduction, device, do_penalty)
        penalty_grads_final = rotate_penalty_grads(penalty_grads_final, loss_grads_final, args.grad_rotate, do_penalty)

        loss_unsplit_grad_scaler, loss_grad_scaler, penalty_grad_scaler, gradnorm_loss, info_dict = \
            calculate_scalers(loss_unsplit_grads_final, loss_grads_final, penalty_grads_final, 
                              loss_unsplit_aggregator,  loss_env,         penalty_env,
                              loss_unsplit_weight,      loss_weight,      penalty_weight,
                              ema,
                              gradnorm_balancer, do_gradnorm,
                              args, do_unsplit_loss, do_loss, do_penalty, device)

        """
        Don't multiply individual task's loss by scaler, since it's misleading
        Only multiply the gradients since this is what determines how tasks' losses are updated
        loss_unsplit_weighted *= loss_unsplit_grad_scaler
        loss_weighted      *= loss_grad_scaler
        penalty_weighted   *= penalty_grad_scaler 
        """
        for pind, p in enumerate(net.parameters()):        
            total_grad_flat_weighted = (   loss_unsplit_grads_final[pind] * loss_unsplit_weight * loss_unsplit_grad_scaler
                                         + loss_grads_final[pind]         * loss_weight         * loss_grad_scaler     
                                         + penalty_grads_final[pind]      * penalty_weight      * penalty_grad_scaler  
                                       )
            if p.grad is None:
                p.grad  = total_grad_flat_weighted.view(p.shape)
            else:
                p.grad += total_grad_flat_weighted.view(p.shape)
        
        # -----------------------
        # Step 3: optimizer step
        # -----------------------
        if (args.penalty_iters > 0) and (epoch == args.penalty_iters) and do_penalty and (not args.increasing_weight):
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            utils.reset_optimizer(train_optimizer)

        train_optimizer.step()
        train_optimizer.zero_grad(set_to_none=True)        # clear gradients at beginning of next gradients batch

        gradnorm_update(gradnorm_balancer, gradnorm_loss, gradnorm_optimizer, args, do_gradnorm)

        # True loss reflecting progress does NOT include balancing scalers
        loss_weighted          = loss_weight         * loss_env.mean()
        loss_unsplit_weighted  = loss_unsplit_weight * loss_unsplit_aggregator.mean()
        penalty_weighted       = penalty_weight      * penalty_env.mean()

        loss_batch_weighted = (loss_unsplit_weighted + # loss_unsplit_aggregator is a scalar normalized over macro-batch
                               penalty_weighted      + # mean over envs normalized over macro-batch
                               loss_weighted           # mean over envs normalized over macro-batch
                              )

        # total loss is sum of losses so far over entire batch aggregation period.
        total_unsplit_loss_weighted += (loss_unsplit_weight * loss_unsplit_aggregator).item() * this_batch_size * gradients_accumulation_steps
        total_irm_loss_weighted  += (penalty_weight   * penalty_env.mean()).item()   * this_batch_size * gradients_accumulation_steps
        total_env_loss_weighted  += (loss_weight      * loss_env.mean()).item()      * this_batch_size * gradients_accumulation_steps
        total_loss_weighted      += loss_batch_weighted.item()                       * this_batch_size * gradients_accumulation_steps
        
        if args.print_batch:
            print() # this causes each tqdm update to be printed on a separare line
        unsplit_str = f'Unsplit/{loss_unsplit_type}' if loss_unsplit_type is not None else 'Unsplit'

        desc_str = f"Epoch [{epoch}/{epochs}] [{trained_samples}/{total_samples}]" + \
                   f" {args.ssl_type}" + \
                   f" Total {total_loss_weighted/trained_samples:.3e}" + \
                   f" {unsplit_str} {total_unsplit_loss_weighted/trained_samples:.3e}" + \
                   f" Env {total_env_loss_weighted/trained_samples:.3e}" + \
                   f" {args.penalty_type} {total_irm_loss_weighted/trained_samples:.3e}" + \
                   f" LR {train_optimizer.param_groups[0]['lr']:.4f} PW {penalty_weight_orig:.6g}" + \
                   f" dot: ll {info_dict['ngl2']:.2e} lk {info_dict['dot_lk']:.2e} lp {info_dict['dot_lp']:.2e} kk {info_dict['ngk2']:.2e}" + \
                   f" kp {info_dict['dot_kp']:.2e} pp {info_dict['ngp2']:.2e}" + \
                   f" cos: lk {info_dict['cos_lk']:.3e} lp {info_dict['cos_lp']:.3e} kp {info_dict['cos_kp']:.2e}" + \
                   f" w/v: k {info_dict['w_k']:.4f}/{info_dict['v_k']:.4f} l {info_dict['w_l']:.4f}/{info_dict['v_l']:.4f}" + \
                   f" p {info_dict['w_p']:.4f}/{info_dict['v_p']:.4f}" + \
                   f" decr: l {info_dict['loss_decrease_cond']:.2e} k {info_dict['loss_unsplit_decrease_cond']:.2e} p {info_dict['penalty_decrease_cond']:.2e}" + \
                   f" gn_loss {info_dict['gradnorm_loss']:.4e} rates: {info_dict['gradnorm_rates_str']} gn_gpm: {info_dict['gn_pm']}" + \
                   f" Lp: cos {info_dict['cos_Lp']:.3e} dot {info_dict['dot_Lp']:.3e} gn_prgrs {info_dict['gradnorm_progress']:.6g}"
        desc_str += loss_module.get_debug_info_str()
        train_bar.set_description(desc_str)

        if (batch_index % 10 - gradients_accumulation_steps + 1) == 0:
           utils.write_log('Train Epoch: [{:d}/{:d}] [{:d}/{:d}] {}: Total: {:.4f} First: {:.4f} Env: {:.4f}'
                            .format(epoch, epochs, trained_samples, total_samples, args.ssl_type, 
                                    total_loss_weighted/trained_samples, 
                                    total_unsplit_loss_weighted/trained_samples, 
                                    total_env_loss_weighted/trained_samples) + 
                            ' {}: {:.4g} LR: {:.4f} PW {:.4f} GN {:.4f}'
                            .format(args.penalty_type, total_irm_loss_weighted/trained_samples, train_optimizer.param_groups[0]['lr'], 
                                    penalty_weight_orig, info_dict['gradnorm_loss']) + 
                            ' dot {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e}'
                            .format(info_dict['ngl2'], info_dict['dot_lk'], info_dict['dot_lp'], info_dict['ngk2'], info_dict['dot_kp'], info_dict['ngp2']) +
                            ' rates {}'
                            .format(info_dict['gradnorm_rates_str']) + 
                            ' decr l {:.2e} k {:.2e} p {:.2e} gn_gpm {} Lp cos {:4f} delta {:.3e}'
                            .format(info_dict['loss_decrease_cond'], info_dict['loss_unsplit_decrease_cond'], info_dict['penalty_decrease_cond'], 
                                    info_dict['gn_pm'], info_dict['cos_Lp'], info_dict['dot_Lp']),
                            log_file=log_file)
                                        
        # Prepare for next iteration
        gradients_accumulation_step = 0
        alternating_gradients_update = 0 if alternating_gradients_update > 0 else 1
        penalty_aggregator.zero_()
        loss_unsplit_aggregator.zero_()
        loss_aggregator.zero_()
        halves_sz.zero_()
        for par in loss_grads: # over list
            par.zero_()
        for par in penalty_grads: # over list
            par.zero_()
        for par in loss_unsplit_grads_final: # over list
            par.zero_()
        del penalty_env, loss_env, loss_batch_weighted
        del info_dict
        torch.cuda.empty_cache()

        loss_module.post_batch()
        if loss_unsplit_module is not None:
            loss_unsplit_module.post_batch()
    # end for batch_index, data_env in enumerate(train_bar):
    return total_loss_weighted / trained_samples

def train_partition(net, update_loader, soft_split, random_init=False, args=None, net_momentum=None, queue=None, **kwargs):

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

    """
    What's the difference between offline and online modes:
    OFFLINE MODE: 
        First extract features from the whole dataset and then run optimization of partitioning over multiple epochs
        with THOSE features frozen.
        The number of times the model is run (in evaluation mode) on the dataset is ONE.
    ONLINE MODE:
        Extract features from a batch and then run optimization of partitioning over multiple epochs/batches.
        Since the model is NOT updated during this it might appear that the features do NOT change. But this is INCORRECT,
        because raw samples are transformed using transforms with random elements. Hence the positives and negatives
        CHANGE over batches / epochs.
        The number of times the model is run (in evaluation mode) on the dataset is NUMBER OF EPOCHS.
    """
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
                desc='train_partition(): Feature extracting'
            )
            if True:
                for pos_, target, Index in train_bar:
                    pos_ = pos_.cuda(non_blocking=True)

                    if transform is not None:
                        pos_1 = transform(pos_)
                        pos_2 = transform(pos_)
                    if target_transform is not None:
                        target = target_transform(target)

                    if args.ssl_type.lower() == 'moco' or args.ssl_type.lower() == 'mocosupcon':
                        feature_1, out_1 = net(pos_1)
                        with torch.no_grad():
                            feature_2, out_2 = model_momentum(pos_2)
                    else:        
                        feature_1, out_1 = net(pos_1)
                        feature_2, out_2 = net(pos_2)
                    feature_bank_1.append(out_1.cpu())
                    feature_bank_2.append(out_2.cpu())
                feature1 = torch.cat(feature_bank_1, 0)
                feature2 = torch.cat(feature_bank_2, 0)
            else:
                feature1 = torch.rand(len(soft_split), 128, dtype=torch.float)
                feature2 = torch.rand(len(soft_split), 128, dtype=torch.float)
            
        updated_split = utils.auto_split_offline(feature1, feature2, soft_split, temperature, args.irm_temp, loss_mode='v2', irm_mode=args.irm_mode,
                                         irm_weight=args.irm_weight_maxim, constrain=args.constrain, cons_relax=args.constrain_relax, nonorm=args.nonorm, 
                                         log_file=log_file, batch_size=uo_bs, num_workers=uo_nw, prefetch_factor=uo_pf, persistent_workers=uo_pw,
                                         ssl_type=args.ssl_type_partition.lower(), queue=queue, dataset=update_loader.dataset)
    else:
        updated_split = utils.auto_split(net, update_loader, soft_split, temperature, args.irm_temp, loss_mode='v2', irm_mode=args.irm_mode,
                                     irm_weight=args.irm_weight_maxim, constrain=args.constrain, cons_relax=args.constrain_relax, 
                                     nonorm=args.nonorm, log_file=log_file)
    np.save("results/{}/{}/{}_{}{}".format(args.dataset, args.name, 'GroupResults', epoch, ".txt"), updated_split.cpu().numpy())
    return updated_split

def get_feature_bank(net, memory_data_loader, args, progress=False, prefix="Test:"):
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
    
        if isinstance(test_data_loader.dataset, Subset):
            dataset = test_data_loader.dataset.dataset
            idcs    = test_data_loader.dataset.indices
        else:
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
        per_class_correct = torch.zeros(c, dtype=torch.long, device=feature_bank[0].device)
        per_class_total   = torch.zeros(c, dtype=torch.long, device=feature_bank[0].device)

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
            one_hot_label = torch.zeros(data.size(0) * args.k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
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
                pred_scores_list.append(pred_scores)

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

    return total_top1 / total_num * 100, total_top5 / total_num * 100, macro_acc * 100
    
def load_checkpoint(path, model, model_momentum, optimizer, gradnorm_balancer, gradnorm_optimizer, device="cuda"):
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
            new_parameters = gradnorm_balancer.load_state_dict(
                checkpoint["state_dict_gradnorm"],
            )
            msg_gradnorm = f'new parameters: {new_parameters}'
        else:
            msg_gradnorm = "no gradnorm in checkpoint"
    else:
        msg_gradnorm = "gradnorm not used"

    # Restore optimizer (if available)
    #FIX ME!!!!!!!!!
    if False and "optimizer" in checkpoint and checkpoint["optimizer"] is not None:

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
    parser = argparse.ArgumentParser(description='Train IP-IRM')
    parser.add_argument('--ssl_type', default='MoCo', type=str, choices=['MoCo', 'SimSiam', 'MoCoSupCon'], help='SSL type')    
    parser.add_argument('--ssl_type_partition', default='MoCoSupCon', type=str, choices=['MoCo', 'SimSiam', 'MoCoSupCon', 'SimCLR'], help='SSL type for partition')    
    parser.add_argument('--penalty_type', default='IRM', type=str, choices=['IRM', 'VREx'], help='Penalty type')        
    parser.add_argument('--penalty_sigma', default=None, type=float, help='Noise level to inject into penalty')        
    parser.add_argument('--grad_rotate', default=None, type=float, nargs=2, help='rotate gradients')      
    parser.add_argument('--loss_unsplit_type', default=None, type=str, choices=['CE', 'CEweighted'], help='Loss unsplit type')    

    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--tau_plus', default=0.1, type=float, help='Positive class priorx')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--knn_temp', default=0.5, type=float, help='Temperature used in KNN softmax')
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
    parser.add_argument('--penalty_unsplit_cont', default=1.0, type=float, help='cont unsplit penalty weight')
    parser.add_argument('--Lscaler', default=1.0, type=float, help='Global scaler for losses gards')
    parser.add_argument('--penalty_iters', default=0, type=int, help='penalty weight start iteration')
    parser.add_argument('--increasing_weight', nargs=5, type=float, default=None, help='increasing penalty weight', 
            metavar='penalty_warmup, scale, speed, eps, debug')
    parser.add_argument('--env_num', default=2, type=int, help='num of the environments')
    parser.add_argument('--weight_env_eps', default=0., type=float, help='eps for per-env grad noise')

    parser.add_argument('--maximize_iter', default=30, type=int, help='when maximize iteration')
    parser.add_argument('--irm_mode', default='v1', type=str, help='irm mode when maximizing')
    parser.add_argument('--irm_weight_maxim', default=1, type=float, help='irm weight in maximizing')
    parser.add_argument('--irm_temp', default=0.5, type=float, help='irm loss temperature')
    parser.add_argument('--random_init', action="store_true", default=False, help='random initialization before every time update?')
    parser.add_argument('--constrain', type=float, default=0., help='weight of contrain to make two envs similar sized')
    parser.add_argument('--constrain_relax', action="store_true", default=False, help='relax the constrain?')
    parser.add_argument('--retain_group', action="store_true", default=False, help='retain the previous group assignments?')
    parser.add_argument('--nonorm', action="store_true", default=False, help='not use norm for contrastive loss when maximizing')
    parser.add_argument('--offline', action="store_true", default=False, help='save feature at the beginning of the maximize?')
    parser.add_argument('--partition_reinit', action="store_true", default=False, help='reinit partition')

    parser.add_argument('--debug', action="store_true", default=False, help='debug?')
    parser.add_argument('--print_batch', action="store_true", default=False, help='print every batch')
    parser.add_argument('--groupnorm', action="store_true", default=False, help='use group contrastive loss?')
    parser.add_argument('--unsplit_cont', action="store_true", default=False, help='unsplit original contrastive?')
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
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--save_root', type=str, default='save', help='root dir for saving')
    parser.add_argument('--checkpoint_freq', default=3, type=int, metavar='N',
                    help='checkpoint epoch freqeuncy')
    parser.add_argument('--val_freq', default=3*3, type=int, metavar='N',
                    help='validation epoch freqeuncy')
    parser.add_argument('--test_freq', default=5*5, type=int, metavar='N',
                    help='test epoch freqeuncy')
    parser.add_argument('--norandgray', action="store_true", default=False, help='skip rand gray transform')
    parser.add_argument('--evaluate', type=str, default=None, nargs="*", choices=['val', 'test'], help='only evaluate')
    parser.add_argument('--extract_features', action="store_true", help="extract features for post processiin during evaluate")
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

    c = len(memory_data.classes) if args.dataset != "ImageNet" else args.class_num
    print('# Classes: {}'.format(c))

    # model setup and optimizer config
    ssl_type = args.ssl_type.lower()
    second_fc = c if args.loss_unsplit_type else None
    if ssl_type == 'moco' or ssl_type == 'mocosupcon':
        model = ModelResnet(feature_dim, image_class=image_class, state_dict=state_dict, second_fc=second_fc).cuda()
    elif ssl_type == 'simsiam':
        model = SimSiam(feature_dim, image_class=image_class, state_dict=state_dict).cuda()
    else:
        raise NotImplemented
    if state_dict is not None:
        print("<= loaded pretrained checkpoint '{}'".format(args.pretrain_path))

    model = nn.DataParallel(model)

    if ssl_type == 'moco' or ssl_type == 'mocosupcon':
        model_momentum = copy.deepcopy(model)
        for p in model_momentum.parameters():
            p.requires_grad = False
        momentum = args.momentum              # momentum for model_momentum
        queue_size = args.queue_size
        queue = FeatureQueue(queue_size, feature_dim, device=device, dtype=torch.float32, indices=True)
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

    if args.opt == "Adam":
        #FIX ME!!!!!!!!!
        #optimizer          = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)
        params = []
        if ssl_type == "simsiam":
            if args.featurizer_lr > 0:
                params.append({'params': model.module.f.parameters(), 'lr': args.featurizer_lr})
            if args.projector_lr > 0:
                params.append({'params': model.module.projector.parameters(), 'lr': args.projector_lr})
            if args.predictor_lr > 0:
                params.append({'params': model.module.predictor.parameters(), 'lr': args.predictor_lr})
        else:
            if args.featurizer_lr > 0:
                params.append({'params': model.module.f.parameters(), 'lr': args.featurizer_lr})
            if args.projector_lr > 0:
                params.append({'params': model.module.g.parameters(), 'lr': args.projector_lr})
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
                load_checkpoint(args.resume, model, model_momentum, optimizer, gradnorm_balancer, gradnorm_optimizer)
            
            # dummy for debug multiple partitions
            # updated_split_all.append(torch.randn((len(update_data), args.env_num), requires_grad=True, device=device))
            if not args.baseline:
                num_partitions = len(updated_split_all) if updated_split_all is not None else 0
                print(f"found {num_partitions} partitions")
                if updated_split_all:
                    if not all([len(s) == len(update_data) for s in updated_split_all]) and args.evaluate is None:
                        print([len(s) for s in updated_split_all], len(update_data))
                        assert False, "Partitons from checkpoint have different length from dataset" 
                    assert updated_split_all[0].size(-1) == args.env_num, \
                        f"env_num in args {args.env_num} doesn't match that in partitions {updated_split_all[0].size(-1)}"
                if (ema_ is not None) and (args.ema == 'retain'): # exists in checkpoint
                    ema = ema_
                ema.set_active(args.ema) # set to what the user has currently set
                # gradnorm restores only attributes needed to continue running. arguments are taken from  user args
                gradnorm_balancer.set_tau(args.gradnorm_tau) # always set tau to currently provided value; also converts None to values

            # use current LR, not the one from checkpoint
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
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
        if args.split_train_for_test:
            mem_data = random_split(memory_data, args.split_train_for_test)
            memory_data = mem_data[0]
        memory_loader = DataLoader(memory_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
            pin_memory=True, persistent_workers=te_pw)
        feauture_bank, feature_labels = get_feature_bank(model, memory_loader, args, progress=True, prefix="Evaluate:")
        if args.split_train_for_test:
            print('eval on train data')
            train_loader = DataLoader(mem_data[1], batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
                pin_memory=True, persistent_workers=te_pw)
            train_acc_1, train_acc_5, train_macro_acc = test(model, feauture_bank, feature_labels, train_loader, args, progress=True, prefix="Train:")
        if len(args.evaluate) == 0:
            args.evaluate = ['val', 'test']
        if 'val' in args.evaluate:
            print('eval on val data')
            val_loader = DataLoader(val_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=True, 
                pin_memory=True, persistent_workers=te_pw)
            val_acc_1, val_acc_5, val_macro_acc = test(model, feauture_bank, feature_labels, val_loader, args, progress=True, prefix="Val:")
        if 'test' in args.evaluate:
            print('eval on test data')
            test_loader = DataLoader(test_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=True, 
                pin_memory=True, persistent_workers=te_pw)
            test_acc_1, test_acc_5, test_macro_acc = test(model, feauture_bank, feature_labels, test_loader, args, progress=True, prefix="Test:")
        exit()

    if not args.resume and os.path.exists(log_file):
        os.remove(log_file)            
    
    if args.gradnorm_rescale_weights:
        gradnorm_balancer.rescale_weights()
    
    if ssl_type == 'moco' or ssl_type == 'mocosupcon':
        kwargs = {'net_momentum': model_momentum, 'queue': queue, 'temperature': temperature, 'momentum': momentum}
    elif ssl_type == 'simsiam':
        kwargs = {}
        
    if args.loss_unsplit_type == 'CEweighted': # weight per-class loss w/ its inverse frequency
        labels = train_data.targets if isinstance(train_data.targets, torch.Tensor) else torch.tensor(train_data.targets)
        labels = target_transform(labels) if target_transform else labels
        counts = torch.bincount(labels)
        class_weights = counts.sum() / counts.float() / args.class_num  # use inverse frequency
        kwargs['CEweights'] = class_weights

    # update partition for the first time, if we need one
    if not args.baseline:
        if (not resumed) or args.partition_reinit or (resumed and (updated_split is None) and ((args.penalty_cont > 0) or (args.penalty_weight > 0))):  
            print("create initial partition")
            if args.dataset != "ImageNet":
                updated_split = torch.randn((len(update_data), args.env_num), requires_grad=True, device=device)
            else:
                updated_split = torch.randn((len(update_data), args.env_num), requires_grad=True, device=device)
                if args.offline:
                    # It is MANDATORY that 'shuffle' is False, otherwise samples won't match their weights
                    upd_loader = DataLoader(update_data, batch_size=u_bs, num_workers=u_nw, prefetch_factor=u_pf, shuffle=False, 
                        drop_last=False, pin_memory=True, persistent_workers=u_pw)
                else:
                    upd_loader = DataLoader(update_data, batch_size=u_bs, num_workers=u_nw, prefetch_factor=u_pf, shuffle=True,
                        drop_last=True, pin_memory=True, persistent_workers=u_pw)
            updated_split = train_partition(model, upd_loader, updated_split, random_init=args.random_init, args=args, **kwargs)
            updated_split_all = [updated_split.clone().detach()]
            assert all([len(s) == len(update_data) for s in updated_split_all]), "Parititon different length from dataset" 
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
                'state_dict_gradnorm':  gradnorm_balancer.state_dict(),
                'gradnorm_optimizer':   gradnorm_optimizer.state_dict(),
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

    for epoch in range(start_epoch, epochs + 1):
        if train_loader is None:
            train_loader = DataLoader(train_data, batch_size=tr_bs, num_workers=tr_nw, prefetch_factor=tr_pf, shuffle=True, 
                                pin_memory=True, persistent_workers=tr_pw, drop_last=tr_dl)

        # Minimize step (Step 1)
        if not args.baseline:
            upd_split = updated_split_all if args.retain_group else updated_split
        else:
            upd_split = None
            updated_split = None
            updated_split_all = None            

        train_loss = train_env(model, train_loader, optimizer, upd_split, tr_bs, args, **kwargs)

        if (epoch % args.maximize_iter == 0) and (not args.baseline):
            # Maximize Step (Step 2)
            train_loader = shutdown_loader(train_loader)
            gc.collect()
            if args.offline:
                # It is MANDATORY that 'shuffle' is False, otherwise samples won't match their weights
                upd_loader = DataLoader(update_data, batch_size=u_bs, num_workers=u_nw, prefetch_factor=u_pf, shuffle=False, 
                    pin_memory=False, persistent_workers=u_pw)
            else:
                upd_loader = DataLoader(update_data, batch_size=u_bs, num_workers=u_nw, prefetch_factor=u_pf, shuffle=True, pin_memory=True, 
                    drop_last=True, persistent_workers=u_pw)
            updated_split = train_partition(model, upd_loader, updated_split, random_init=args.random_init, args=args, **kwargs)
            upd_loader = shutdown_loader(upd_loader)
            gc.collect()              # run Python's garbage collector
            updated_split_all.append(updated_split)
            assert all([len(s) == len(update_data) for s in updated_split_all]), "Parititons different length from dataset" 
            # reset optimizer after new split created
            utils.reset_optimizer(optimizer)

        feature_bank, feature_labels = None, None
        # eval knn every test_freq/val_freq and last epochs
        if (                                 (epoch >= args.test_freq) and ((epoch % args.test_freq == 0) or (epoch == epochs))) or \
           ((args.dataset == 'ImageNet') and (epoch >= args.val_freq)  and ((epoch % args.val_freq == 0)  or (epoch == epochs))):
            if train_loader is not None:
                train_loader = shutdown_loader(train_loader)
                gc.collect()
            memory_loader = DataLoader(memory_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=False, 
                pin_memory=False, persistent_workers=te_pw)
            feauture_bank, feature_labels = get_feature_bank(model, memory_loader, args, progress=True, prefix="Evaluate:")
            memory_loader = shutdown_loader(memory_loader)
            gc.collect()              # run Python's garbage collector

        if (epoch >= args.test_freq) and ((epoch % args.test_freq == 0) or (epoch == epochs)): # eval knn every test_freq epochs
            test_loader = DataLoader(test_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=True, 
                pin_memory=False, persistent_workers=te_pw)
            test_acc_1, test_acc_5, test_macro_acc = test(model, feauture_bank, feature_labels, test_loader, args, progress=True, prefix="Test:")
            test_loader = shutdown_loader(test_loader)
            gc.collect()              # run Python's garbage collector
            txt_write = open("results/{}/{}/{}".format(args.dataset, args.name, 'knn_result.txt'), 'a')
            txt_write.write('\ntest_acc@1: {}, test_acc@5: {}, test_macro_acc: {}'.format(test_acc_1, test_acc_5, test_macro_acc))
            torch.save(model.state_dict(), 'results/{}/{}/model_{}.pth'.format(args.dataset, args.name, epoch))

        if (epoch >= args.val_freq) and ((epoch % args.val_freq == 0) or (epoch == epochs)) and (args.dataset == 'ImageNet'):
            # evaluate on validation set
            val_loader = DataLoader(val_data, batch_size=te_bs, num_workers=te_nw, prefetch_factor=te_pf, shuffle=True, 
                pin_memory=False, persistent_workers=te_pw)
            acc1, _, _ = test(model, feauture_bank, feature_labels, val_loader, args, progress=True, prefix="Val:")
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
