import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm

import utils

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

