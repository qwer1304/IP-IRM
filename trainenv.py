import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm

import utils

from math import ceil, prod

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

    def pre_micro_batch(self, features_1, features_2, **kwargs):
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

    def filter_indices(self, indices, **kwargs):
        return indices

# ---------------------------
# MoCo+SupCon Loss Module
# ---------------------------
class MoCoSupConLossModule(LossModule):
    def __init__(self, *args, net_momentum=None, queue=None, temperature=None, debug=False, filter_indices=None, **kwargs):
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
        self.filter_indices_hook = filter_indices
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

    def pre_micro_batch(self, pos_q, pos_k, indexs=None, normalize=True, dataset=None, **kwargs):
        # 'indexs' are batch samples indices in the dataset
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
        assert len(pos_q) == len(indexs), f"len(pos_q) {len(pos_q)} != len(indexs) {len(indexs)}"
        assert len(pos_q) == len(pos_k), f"len(pos_q) {len(pos_q)} != len(pos_k) {len(pos_k)}"

        out_q = self.net.module.g(pos_q)
        if normalize:
            out_q = F.normalize(out_q, dim=1)
        with torch.no_grad():
            out_k = self.net_momentum.module.g(pos_k)
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
        y_batch = get_targets(indexs, dataset, pos_q.device)
        y_queue = get_targets(idx_queue, dataset, pos_q.device)
        y_all = torch.cat([y_batch, y_queue], dim=0) # (N,)

        logits = (out_q @ k_all.T) / self.temperature # (B,N)
        
        # for each sample in the batch (row) give the samples in the batch and queue w/ the same label
        pos_mask = (y_batch[:, None] == y_all[None, :])   # (B,N)
        pos_mask[:, :len(y_batch)].fill_diagonal_(False)  # remove self-keys
        num_pos = pos_mask.sum(dim=1, keepdim=True).clamp(min=1)

        # Replace non-positives with -inf
        pos_logits = logits.masked_fill(~pos_mask, -1e9)
        # One logit per anchor = logsumexp over positives
        l_pos = torch.logsumexp(pos_logits, dim=1, keepdim=True) #- num_pos.log() # (B,1)
        
        l_neg = logits.masked_fill(pos_mask, -1e9) # (B,N)
        
        self._logits = torch.cat([l_pos, l_neg], dim=1) # (B,1+N), positive logit is in column 0
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
        # 'idxs' are indices into the batch
        # 'p', 'env' select the indices into the queue of the relevant samples
        if idxs is None:
            idxs = torch.arange(self._logits.size(0), device=self._logits.device)
        l_pos = self.l_pos # (B,1), positive logit for the whole micro-batch
        l_neg = self.l_neg # (B,N), negative logits for the whole micro-batch
        if p is not None:
            l_neg = l_neg[:, self.neg_idxs[p][env]] # scope negative logits to (p,env)-samples
        self._logits = torch.cat([l_pos, l_neg], dim=1) # (B,1+N')
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

    def filter_indices(self, indices, **kwargs):
        return self.filter_indices_hook(indices, **kwargs) if self.filter_indices_hook is not None else super().filter_indices(indices, **kwargs)
        
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

    def pre_micro_batch(self, pos_q, pos_k, indexs=None, normalize=True, **kwargs):
        assert indexs is not None, 'indexs cannot be None'
        assert len(pos_q) == len(indexs), f"len(pos_q) {len(pos_q)} != len(indexs) {len(indexs)}"
        assert len(pos_q) == len(pos_k), f"len(pos_q) {len(pos_q)} != len(pos_k) {len(pos_k)}"

        out_q = self.net.module.g(pos_q)
        if normalize:
            out_q = F.normalize(out_q, dim=1)
        with torch.no_grad():
            out_k = self.net_momentum.module.g(pos_k)
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

    def pre_micro_batch(self, x1, x2, **kwargs):
        # Unnormalized!
        z1 = self.net.module.g(x1)
        z2 = self.net.module.g(x2)
        p1 = self.net.module.h(z1)
        p2 = self.net.module.h(z2)
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

    def pre_micro_batch(self, x1, x2, normalize=True, labels=None, weights=None, **kwargs):
        x = torch.cat([x1, x2], dim=0)
        out = self.net.module.fc(x)
        if normalize:
            out = F.normalize(out, dim=1)
        self._logits = out
        self.labels = torch.cat([labels, labels], dim=0)
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

def calculate_penalty_grads_final(penalty_grads, penalty_aggregator, penalty_weight_env, halves_sz, penalty_calculator, penalty_sigma, reduction, device, do_penalty):
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
                    sigma=penalty_sigma,
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
                                 [g], [loss_grads_final[pind]], theta=theta)[0].clone() 
                               for pind, g in enumerate(penalty_grads_final)
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
            grads_weighted        = [g.detach().clone() * weight * Lscaler for g in grads_final if g is not None]
            grads_weighted_vector = torch.cat([g for g in grads_weighted]) 
            grads_norm_weighted   = grads_weighted_vector.norm()
        else:
            grads_weighted = [torch.zeros_like(g) for g in grads_final if g is not None]
            assert default_grads_weighted_vector is not None, "default_grads_weighted_vector not given when do_flag is False"
            grads_weighted_vector = default_grads_weighted_vector
            grads_norm_weighted   = torch.tensor(0., dtype=torch.float, device=device)
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
    cos_lp   = delta_lp / (loss_grad_norm_weighted         * penalty_grad_norm_weighted + 1e-12)
    cos_kp   = delta_kp / (loss_unsplit_grad_norm_weighted * penalty_grad_norm_weighted + 1e-12)

    Loss_grads_flat_weighted = [loss_unsplit_grads_final_weighted[p] + loss_grads_final_weighted[p] for p in range(len(loss_grads_final_weighted))]
    L_grads_flat_weighted    = l_unsplit_grads_flat_weighted + l_grads_flat_weighted
    cos_Lp                   = F.cosine_similarity(L_grads_flat_weighted, p_grads_flat_weighted, dim=0)
    dot_Lp                   = L_grads_flat_weighted.dot(p_grads_flat_weighted)

    loss_weighted         = loss_weight         * loss_env.mean()
    loss_unsplit_weighted = loss_unsplit_weight * loss_unsplit_aggregator.mean()
    penalty_weighted      = penalty_weight      * penalty_env.mean()

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
        losses_dict['penalty']          = penalty_weighted
        grad_norms_dict['penalty']      = ngp
    if do_loss:
        losses_dict['loss']             = loss_weighted
        grad_norms_dict['loss']         = ngl
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
        'v_l':               gradnorm_balancer.task_weights['loss'].item()         if 'loss'         in gradnorm_balancer.task_weights else 0.,
        'v_p':               gradnorm_balancer.task_weights['penalty'].item()      if 'penalty'      in gradnorm_balancer.task_weights else 0.,
        'gn_pm':             gn_pm,
        # if cond > 0, the corresponding quantity would decrease
        'loss_decrease_cond':         loss_grad_scaler         * ngl2 + loss_unsplit_grad_scaler*dot_lk + penalty_grad_scaler*dot_lp,
        'loss_unsplit_decrease_cond': loss_unsplit_grad_scaler * ngk2 + loss_grad_scaler*dot_lk         + penalty_grad_scaler*dot_kp,
        'penalty_decrease_cond':      penalty_grad_scaler      * ngl2 + loss_unsplit_grad_scaler*dot_kp + loss_grad_scaler*dot_lp,
        'gradnorm_rates_str':         " ".join([f'{n} {r:.4f}' for n,r in zip([task_names_2_klp[k] for k in task_names], gradnorm_rates)]) if do_gradnorm else "",  
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
    
def set_BN_adapt(net, adapt_bn, bn_momentum):
    for m in net.modules():
        if isinstance(m, (torch.nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
            if not adapt_bn:
                m.eval()                 # use stored running stats
                m.track_running_stats = True
            else:
                m.train()                 # learn pars
                m.track_running_stats = True
                m.momentum = bn_momentum

def calculate_grads(loss, net, retain_graph=False):
    grads = torch.autograd.grad(
        outputs=loss,
        inputs=tuple(net.parameters()),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return grads

def get_stochastic_partitions(all_partitions, k=5):
    # all_partitions a list of tensors
    num_total = len(all_partitions)
    # Pick k random indices to keep
    active_idxs = sorted(torch.randperm(num_total)[:k].tolist())
    
    # Create list: tensor of index that was picked
    subset = [all_partitions[i] for i in active_idxs]
    
    return subset, active_idxs

def print_grads(grads, net, prefix=""):
    for pind, (name, p) in enumerate(net.named_parameters()):
        if grads[pind] is None:
            continue
        print(f"{prefix} pind {pind} name {name} norm {grads[pind].norm():.2e}")
        
# ssl training with IP-IRM
def train_env(net, train_loader, train_optimizer, partitions, batch_size, epoch, args, **kwargs):

    ema = kwargs['ema']
    gradnorm_balancer, gradnorm_optimizer = kwargs['gradnorm_balancer'], kwargs['gradnorm_optimizer']
    log_file = kwargs['log_file']
    
    net.train()
    set_BN_adapt(net, args.adapt_bn, args.bn_momentum)

    if isinstance(partitions, list): # if retain previous partitions
        assert args.retain_group
    else:
        partitions = [partitions]
    if args.decimate_partitions:
        assert args.decimate_partitions <= len(partitions), f"# of partitions to decimate {args.decimate_partitions} > # partitions {len(partitions)}"
        num_partitions = args.decimate_partitions
    else:    
        num_partitions = len(partitions) # some partitions may become None subsequent to decimation application
    
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
    total_samples                = len(train_loader.dataset)
    
    trained_samples             = 0
    total_unsplit_loss_weighted = 0.0
    total_env_loss_weighted     = 0.0
    total_pen_loss_weighted     = 0.0
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
    loss_type         = getattr(args, 'ssl_type', 'moco').lower()
    penalty_type      = getattr(args, 'penalty_type', 'irm').lower()
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

    for batch_index, data_env in enumerate(train_bar):

        if args.decimate_partitions:
            assert args.decimate_partitions <= len(partitions), f"# of partitions to decimate {args.decimate_partitions} > # partitions {len(partitions)}"
            partitions, active_partition_idx = get_stochastic_partitions(partitions, k=args.decimate_partitions)
        else:
            active_partition_idx = list(range(num_partitions))
        
        reduction = 'sum' if is_per_env else 'none' # make sure it's the correct one

        data_batch, labels_batch, indexs_batch = data_env # 'data_batch' is an batch of images, 'indexs_batch' is their corresponding indices 
        this_batch_size = len(indexs_batch) # for the case drop_last=False
        
        loss_module.pre_batch(data_batch, indexs_batch, partitions)
        if loss_unsplit_module is not None:
            loss_unsplit_module.pre_batch(data_batch) # weights handled below

        # -----------------------
        # Step 0: micro-batches
        # -----------------------
        mb_list = list(utils.microbatches([data_batch, labels_batch, indexs_batch], gpu_batch_size))

        for j in range(num_halves): # over halves of micro-batches
            for i in [i_ for i_ in range(len(mb_list)) if i_ % num_halves == j]: # loop over micro-batches
                losses_samples_all, losses_samples, penalties_samples, penalty = None, None, None, None
                
                # per micro-batch pipeline
                batch_micro, labels, indexs = mb_list[i]
                if (loss_unsplit_module is not None) and ('CEweights' in kwargs) and (kwargs['CEweights'] is not None):
                    weights         = kwargs['CEweights'] # weights are PER class weights
                else:
                    weights         = None
                batch_micro         = batch_micro.cuda(non_blocking=True)
                labels              = labels.cuda(non_blocking=True)
                indexs              = indexs.cuda(non_blocking=True)
                weights             = weights.cuda(non_blocking=True) if weights is not None else None

                num_samples           = 1 if is_per_env else len(batch_micro)
                num_grads_per_env     = int(not args.baseline) * (int(loss_weight>0) + int(penalty_weight>0)) # 0, 1 or 2
                num_baseline_grads    = int(loss_unsplit_weight>0) * int(args.unsplit_cont) # 0 or 1                                  
                num_grads_per_sample  = max(num_grads_per_env, num_baseline_grads) # max is for the case 'num_grads_per_env' is 0
                num_grads             = num_partitions * args.env_num * num_grads_per_env + num_baseline_grads # number of rows
                if not is_per_env:
                    # for per-sample loss_cont, the number of columns is the number of samples times the number of repeats. 
                    # loss_cont and loss_unsplit share the SAME columns, if loss_unsplit is similar to loss_cont; otherwise loss_unsplit has its own extra column
                    if num_baseline_grads and loss_unsplit_module is None:
                        grad_outputs      = torch.zeros((num_grads, num_samples*num_grads_per_sample), dtype=torch.float, device=device) 
                    elif num_baseline_grads and loss_unsplit_module is not None:
                        grad_outputs      = torch.zeros((num_grads, num_samples*num_grads_per_env + 1), dtype=torch.float, device=device) 
                    else:
                        grad_outputs      = torch.zeros((num_grads, num_samples*num_grads_per_sample), dtype=torch.float, device=device) 
                differentiate_this    = []
                """
                grads_all: 
                    Each row holds gradients for a task
                    First half hold gradients of per-env cont losses, if requested
                    Second half holds gradients for per-env penalties, if requested
                    Last row holds gradients for unsplit loss, if requested
                    Each entry is a tuple with an entry for per net's parameter grad or None
                """
                grads_all             = [None] * num_grads

                """
                prepare for micro-batch in loss-sepcific way:
                    MoCo:    generate two views, get their embeddings from respective encoders, normalize them, etc
                    SimSiam: generate two views, get their projections and predictions, etc
                """
                features_1, features_2 = net.module.f(transform(batch_micro)), net.module.f(transform(batch_micro))
                del batch_micro
                torch.cuda.empty_cache()
                
                # if want unsplit loss w/ its own loss method
                if do_unsplit_loss and (loss_unsplit_module is not None):
                    loss_unsplit_module.pre_micro_batch(features_1, features_2, indexs=indexs, labels=labels, normalize=False,
                        dataset=train_loader.dataset, weights=weights)
                    losses_samples_all = loss_unsplit_module.compute_loss_micro(reduction='sum')
                    if is_per_env:
                        loss = losses_samples_all / 2 / this_batch_size / gradients_accumulation_steps # CE uses both views
                        grads_all[-1] = calculate_grads(loss, net, retain_graph=True)
                    else:
                        # Must be first to be in 1st column 
                        differentiate_this.append(losses_samples_all)

                # mask: assume user sets mask and args.backbone_propagate properly:
                # when it's not needed it's set to 'ident' and args.backbone_propagate==True
                # must apply mask only after unsplit loss has been applied
                if args.backbone_propagate:
                    features_1, features_2 = net.module.mask(features_1), net.module.mask(features_2) 
                else:
                    features_1, features_2 = net.module.mask(features_1.detach()), net.module.mask(features_2.detach()) 

                loss_module.pre_micro_batch(features_1, features_2, indexs=indexs, normalize=(loss_type != 'supcon'), dataset=train_loader.dataset)
                
                # Even if 'do_loss'==False, when SAME loss is used for BOTH loss_cont and loss_unsplit, 'reduction' reflects the correct reduction
                # If SAME loss is used for unsplit and cont losses and any of them has been requested and this is a non-per-env loss
                if not is_per_env:
                    if (do_unsplit_loss and (loss_unsplit_module is None)) or do_loss:
                        # compute unnormalized WHOLE micro-batch loss, no split into envs
                        losses_samples = loss_module.compute_loss_micro(reduction=reduction)
                        differentiate_this.append(losses_samples)
                        if loss_unsplit_module is None:
                            losses_samples_all = losses_samples.sum()

                    if do_penalty:
                        penalties_samples = penalty_calculator.penalty(losses_samples, reduction=reduction)
                        differentiate_this.append(penalties_samples)

                if do_loss or do_penalty:
                    for partition_num, partition in enumerate(partitions):
                        for env in range(args.env_num):
                        
                            is_last = (partition_num == (num_partitions-1)) and (env == (args.env_num-1))

                            # split mb: 'idxs' are indices into 'indexs' that correspond to domain 'env' in 'partition'
                            # 'indexs' are the indices in dataset of samples which are in this micro-batch
                            # For EqInv each partition corresponds to a class. Each env holds positive AND negative samples.
                            # Need to filter the samples s.t. the samples in micro-batch are ONLY those which class==partition
                            # Use 'assign_idxs_multi' to break ties correctly
                            idxs = utils.assign_idxs_multi(indexs, partition, env)
                            idxs = loss_module.filter_indices(idxs, labels=labels[idxs], partition=active_partition_idx[partition_num], env=env)

                            if (N := len(idxs)) == 0:
                                if is_per_env:
                                    assert len(grads_all) > 0, f"env ({partition_num},{env}) has no samples and we don't know the grads shape yet"
                                    if do_loss:
                                        grads_all.append(tuple([g.detach() if g is not None else None for g in grads_all[0]])) # dummy loss's grads
                                    if do_penalty:
                                        grads_all.append(tuple([g.detach() if g is not None else None for g in grads_all[0]])) # dummy penalty's grads
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
                                    grads_all[partition_num*args.env_num + env] = calculate_grads(losses_samples, net, retain_graph=(not is_last) or do_penalty)
                                    loss = losses_samples.detach()
                                else:
                                    # For 'is_per_env'==False, convert per-sample losses to a sum
                                    loss = losses_samples[idxs].sum(dim=0).detach()
                                loss_aggregator[j,partition_num,env] += loss # unnormalized, before penalty scaler
                            # penalties - penalties are ALWAYS a scalar
                            if do_penalty:
                                if is_per_env:
                                    penalties_samples = penalty_calculator.penalty(losses_samples, reduction=reduction)
                                    grads_all[partition_num*args.env_num + env] = calculate_grads(losses_samples, net, retain_graph=not is_last)
                                    penalty = penalties_samples.detach()
                                else:
                                    penalty = penalties_samples[idxs].sum(dim=0).detach()
                                penalty_aggregator[j,partition_num,env] += penalty # unnormalized penalty components before penalty scaler

                            if not is_per_env:
                                # gradients
                                """
                                For non-per-env losses (e.g., SimSiam):
                                    'grad_outputs' is a table w/ each row corresponding to a loss/penalty of an env in a partition.
                                    The top half coresponds to cont losses; the bottom half corresponds to penalties.
                                    The last row corresponds to loss_unsplit.
                                    Each column corresponds to a sample loss/penalty.
                                    The left half corresponds to losses; the right half corresponds to penalties.
                                    The 1st to 'num_samples' columns correspond to loss_unsplit if requested.                                
                                    'grad_outputs[i,j]' is a multiplier of the [i,j]-th entry to differentiate.
                                """
                                # 1. Calculate base indices once
                                # Using standard Python ints for indexing is faster than creating 0-d Tensors
                                base_idx   = partition_num  * args.env_num + env
                                row_stride = num_partitions * args.env_num

                                # 2. Determine Mask and Initial Offset
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
                    # loss - loss is ALWAYS a scalar
                    # CE is computed for BOTH views concatenated
                    num_views = 2 if loss_unsplit_module is not None else 1
                    loss = losses_samples_all.detach()  / num_views / this_batch_size / gradients_accumulation_steps
                    # compute unnormalized gradients for this loss
                    # grad_outputs: one per sample
                    loss_unsplit_aggregator += loss # before scaler

                    if not is_per_env:
                        offset = 0 # 1st column
                        number_of_columns = num_samples if loss_unsplit_module is None else 1
                        grad_outputs[-1][offset:offset+number_of_columns]  = 1.0 / this_batch_size / gradients_accumulation_steps # unweighted

                if is_per_env:
                    """
                    print()
                    print(f"num_samples {num_samples}, num_grads_per_env {num_grads_per_env}, num_baseline_grads {num_baseline_grads}, " +                                  
                          f"num_grads_per_sample {num_grads_per_sample}, num_grads {num_grads}, number_of_columns {number_of_columns}, " + 
                          f"grads_all {len(grads_all)} grads_all[0] {grads_all[0].size()}")
                    """
                    pass
                else:
                    differentiate_this = [t.reshape(-1) for t in differentiate_this] # ensure common shape of 1D tensors
                    differentiate_this = torch.cat(differentiate_this, dim=0) # cat losses and penalties into a single vector length 2B

                    """
                    print()
                    print(f"num_samples {num_samples}, num_grads_per_env {num_grads_per_env}, num_baseline_grads {num_baseline_grads}, " +                                  
                          f"num_grads_per_sample {num_grads_per_sample}, num_grads {num_grads}, number_of_columns {number_of_columns}, " + 
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

                    def convert_batched_to_list_mapped(grads_all_batched):
                        """
                        Converts: Tuple of Tensors [Batch, *Shape]
                        To: List of Tuples arranged as [Original_Last, Original_0, Original_1, ...]
                        """
                        # 1. Determine batch size
                        num_items = 0
                        for g in grads_all_batched:
                            if g is not None:
                                num_items = g.shape[0]
                                break

                        if num_items == 0:
                            return []

                        grads_per_loss = [None] * num_items

                        # 2. Map indices: 
                        for i in range(num_items):
                            current_loss_params = []
                            for g in grads_all_batched:
                                if g is None:
                                    current_loss_params.append(None)
                                else:
                                    # Slicing the i-th batch row
                                    current_loss_params.append(g[i].detach().clone())

                            grads_per_loss[i] = tuple(current_loss_params)

                        return grads_per_loss

                    # 'grads_all' must be a list of gradients per loss.
                    grads_all = convert_batched_to_list_mapped(grads_all)
                #end if is_per_env:

                # update grads' aggregators
                # 1. Setup metadata
                num_env_tasks = (num_grads - num_baseline_grads) // max(1, num_grads_per_env)
                penalty_offset = num_partitions * args.env_num * int(do_loss)
                view_shape = (num_partitions, args.env_num, -1)

                # 2. Consume the list of gradients sample-by-sample
                # This is better for memory because we can clear each sample after processing
                for ii in range(num_grads):
                    # current_grads is the tuple of all parameter grads for sample 'i'
                    current_grads = grads_all[ii]

                    for param_idx, g in enumerate(current_grads):
                        if g is None:
                            continue

                        # Flatten the parameter gradient: (*Param_Dims) -> (Flattened_Dim)
                        g_flat = g.detach().reshape(-1)

                        # 1. Unsplit Loss (Original index 0)
                        if ii == num_grads-1 and do_unsplit_loss:
                            loss_unsplit_grads_final[param_idx] += g_flat

                        # 2. Loss Tasks (Original indices 1 to num_env_tasks)
                        elif 0 <= ii < num_env_tasks and do_loss:
                            k = ii 
                            p = k // args.env_num
                            e = k % args.env_num
                            # Adding directly into the aggregator grid
                            loss_grads[param_idx][j][p, e] += g_flat

                        # 3. Penalty Tasks (Original indices starting at offset+1)
                        elif penalty_offset <= ii < (penalty_offset + num_env_tasks) and do_penalty:
                            k = ii - penalty_offset
                            p = k // args.env_num
                            e = k % args.env_num
                            penalty_grads[param_idx][j][p, e] += g_flat

                # end if not args.baseline:

                loss_module.post_micro_batch()
                loss_module.prepare_for_free()
                if loss_unsplit_module is not None:
                    loss_unsplit_module.post_micro_batch()
                    loss_unsplit_module.prepare_for_free()
                
                # free memory of micro-batch
                del features_1, features_2, indexs, g_flat, g, grads_all, differentiate_this
                if loss is not None: del loss
                if losses_samples_all is not None: del losses_samples_all
                if losses_samples is not None: del losses_samples
                if penalties_samples is not None: del penalties_samples
                if penalty is not None: del penalty
            # end for i in [i_ for i_ in range(len(mb_list)) if i_ % 2 == j]:
            torch.cuda.empty_cache()
        # end for j in range(idxs):
        torch.cuda.empty_cache()
        trained_samples += this_batch_size # total number of samples processed so far
        
        gradients_accumulation_step += 1
        if gradients_accumulation_step < gradients_accumulation_steps:
            continue
        
        loss_weight_env    = make_rand_dither_weight(num_partitions, args.env_num, args.weight_env_eps, device)
        penalty_weight_env = make_rand_dither_weight(num_partitions, args.env_num, args.weight_env_eps, device)

        if do_loss:
            partition_sz = halves_sz.sum(dim=0, keepdim=True) # (1,J,K) # sizes of envs in macro-batch
            loss_env     = loss_aggregator.sum(dim=0, keepdim=True) / (partition_sz+1e-12)  # per env for macro-batch, normalized per env, unweighted
        else:
            loss_env     = torch.tensor(0, dtype=torch.float, device=device)
        if do_penalty:
            penalty_env  = penalty_calculator.penalty_finalize(penalty_aggregator, halves_sz) # normalized per env for macro-batch, unweighted
        else:
            penalty_env  = torch.tensor(0, dtype=torch.float, device=device)

        # Environments gradients
        loss_grads_final = calculate_loss_grads_final(loss_grads, loss_env, loss_weight_env, halves_sz, loss_module, reduction, device, do_loss)

        penalty_grads_final = calculate_penalty_grads_final(penalty_grads, penalty_aggregator, penalty_weight_env, halves_sz, penalty_calculator, 
                                args.penalty_sigma, reduction, device, do_penalty)
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
        print()
        #for pind, p in enumerate(net.parameters()):        
        for pind, (name, p) in enumerate(net.named_parameters()):
            total_grad_flat_weighted = (   loss_unsplit_grads_final[pind] * loss_unsplit_weight * loss_unsplit_grad_scaler
                                         + loss_grads_final[pind]         * loss_weight         * loss_grad_scaler     
                                         + penalty_grads_final[pind]      * penalty_weight      * penalty_grad_scaler  
                                       )
            # Logic to identify the source
            if "f." in name:
                label = "BACKBONE"
            elif "mask" in name:
                label = "MASK"
            elif "projector" in name or "predictor" or "projection" in name:
                label = "CONTRASTIVE_ARM"
            elif "classifier" in name:
                label = "CE_ARM"
            else:
                label = "OTHER"
            print(f"pind {pind}, label {label}, name {name}, ngk {loss_unsplit_grads_final[pind].norm():.2e}, ngl {loss_grads_final[pind].norm():.2e}, ngp {penalty_grads_final[pind].norm():.2e}")
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
        train_optimizer.zero_grad(set_to_none=True)    # clear gradients at beginning of next gradients batch

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
        total_pen_loss_weighted     += (penalty_weight      * penalty_env.mean()).item()      * this_batch_size * gradients_accumulation_steps
        total_env_loss_weighted     += (loss_weight         * loss_env.mean()).item()         * this_batch_size * gradients_accumulation_steps
        total_loss_weighted         += loss_batch_weighted.item()                             * this_batch_size * gradients_accumulation_steps
        
        if args.print_batch:
            print() # this causes each tqdm update to be printed on a separare line
        unsplit_str = f'Unsplit/{loss_unsplit_type}' if loss_unsplit_type is not None else 'Unsplit'

        desc_str = f"Epoch [{epoch}/{args.epochs}] [{trained_samples}/{total_samples}]" + \
                   f" {args.ssl_type}" + \
                   f" Total {total_loss_weighted/trained_samples:.3e}" + \
                   f" {unsplit_str} {total_unsplit_loss_weighted/trained_samples:.3e}" + \
                   f" Env {total_env_loss_weighted/trained_samples:.3e}" + \
                   f" {args.penalty_type} {total_pen_loss_weighted/trained_samples:.3e}" + \
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
                            .format(epoch, args.epochs, trained_samples, total_samples, args.ssl_type, 
                                    total_loss_weighted/trained_samples, 
                                    total_unsplit_loss_weighted/trained_samples, 
                                    total_env_loss_weighted/trained_samples) + 
                            ' {}: {:.4g} LR: {:.4f} PW {:.4f} GN {:.4f}'
                            .format(args.penalty_type, total_pen_loss_weighted/trained_samples, train_optimizer.param_groups[0]['lr'], 
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

