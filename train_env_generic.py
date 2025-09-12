import torch
import torch.nn.functional as F
from torch import optim
from math import ceil, prod
from tqdm import tqdm
import utils  # your existing utility functions

# ---------------------------
# Base IRM Calculator
# ---------------------------
class IRMCalculator:
    """
    Base class for IRM calculation. Subclass this and implement
    `gradients_for_half` to return g_i for a half-batch.
    """
    def __init__(self, irm_temp=1.0):
        self.irm_temp = irm_temp

    def gradients_for_half(self, outputs, targets):
        """
        Compute gradient w.r.t. scale for a half-batch.
        Must be implemented by subclass.
        Returns a tensor g_i.
        """
        raise NotImplementedError

    def macro_batch_loss(self, g_sums, half_split_sz):
        """
        Combine two halves to produce macro-batch IRM penalty.
        g_sums: [2, num_splits, num_envs] (sum of gradients per half)
        half_split_sz: [2, num_splits, num_envs] (sizes of halves)
        Returns IRM loss scalar
        """
        gs0 = g_sums[0] / half_split_sz[0]
        gs1 = g_sums[1] / half_split_sz[1]
        irm_per_env = gs0 * gs1
        return irm_per_env.mean()


# ---------------------------
# CE-based IRM (for MoCo)
# ---------------------------
class CE_IRMCalculator(IRMCalculator):
    def gradients_for_half(self, logits, targets):
        scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
        loss = F.cross_entropy(scale * logits, targets, reduction='sum')
        g_i = torch.autograd.grad(loss, scale, create_graph=True)[0]
        return g_i


# ---------------------------
# Base Loss Module
# ---------------------------
class LossModule:
    """
    Base class for pluggable loss module.
    Subclass for MoCo, SimSiam, etc.
    """
    def pre_micro_batch(self, batch_data):
        """
        Optional: do any pre-processing (e.g., MoCo queue)
        """
        return batch_data

    def compute_loss_micro(self, batch_data):
        """
        Compute micro-batch loss and gradients for IRM.
        Returns:
            - loss_keep_cont (scalar, optional)
            - loss_cont_sum (per split/env sum)
            - g_i (IRM gradient per split/env)
        """
        raise NotImplementedError

    def post_micro_batch(self, outputs):
        """
        Optional: update buffers, e.g., MoCo queue
        """
        pass


# ---------------------------
# MoCo Loss Module
# ---------------------------
class MoCoLossModule(LossModule):
    def __init__(self, net, net_momentum, queue, temperature):
        self.net = net
        self.net_momentum = net_momentum
        self.queue = queue
        self.temperature = temperature

    def pre_micro_batch(self, batch_data):
        # Advance queue read pointer if needed
        batch_size = batch_data[0].size(0)
        self.queue.get(batch_size)
        return batch_data

    def compute_loss_micro(self, batch_data):
        pos_all_batch, indexs_batch = batch_data
        pos_q = pos_all_batch
        pos_k = pos_all_batch  # Simulate augmentation
        _, out_q = self.net(pos_q)
        out_q = F.normalize(out_q, dim=1)
        with torch.no_grad():
            _, out_k = self.net_momentum(pos_k)
            out_k = F.normalize(out_k, dim=1)
        l_pos = torch.sum(out_q * out_k, dim=1, keepdim=True)
        l_neg = torch.matmul(out_q, self.queue.get(self.queue.queue_size - len(indexs_batch), advance=False).t())
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_cont = logits / self.temperature
        labels_cont = torch.zeros(logits_cont.size(0), dtype=torch.long, device=logits.device)
        # sum over batch, per env handled by driver
        loss_keep_cont = F.cross_entropy(logits_cont, labels_cont, reduction='sum')
        return loss_keep_cont.detach(), logits_cont, labels_cont

    def post_micro_batch(self, outputs):
        _, out_k = outputs
        self.queue.update(out_k.detach())


# ---------------------------
# SimSiam Loss Module
# ---------------------------
class SimSiamLossModule(LossModule):
    def __init__(self, net, projector):
        self.net = net
        self.projector = projector  # optional projector if used

    def compute_loss_micro(self, batch_data):
        pos_all_batch, indexs_batch = batch_data
        # compute SimSiam loss
        z1, z2 = self.net(pos_all_batch), self.net(pos_all_batch)  # placeholder
        # simple symmetric negative cosine loss
        p1, p2 = self.projector(z1), self.projector(z2)
        loss = F.mse_loss(p1, z2.detach()) / 2 + F.mse_loss(p2, z1.detach()) / 2
        return loss.detach(), z1, z2


# ---------------------------
# Training Driver
# ---------------------------
def train_env(net, loss_module, irm_calculator, train_loader, train_optimizer, args, epoch, epochs):
    device = next(net.parameters()).device
    net.train()
    loader_batch_size = args.batch_size
    gpu_batch_size = args.micro_batch_size
    gradients_accum_steps = args.gradients_accumulation_batch_size // loader_batch_size
    gpu_accum_steps = ceil(loader_batch_size / gpu_batch_size)
    trained_samples = 0

    bar_format = '{l_bar}{bar:' + str(args.bar) + '}{r_bar}'
    train_bar = tqdm(train_loader, total=len(train_loader), ncols=args.ncols, dynamic_ncols=False, bar_format=bar_format)

    # Buffers for IRM and cont losses
    num_splits = len(args.updated_split)
    num_envs = args.env_num
    half_split_sz = torch.zeros((2, num_splits, num_envs), dtype=torch.float, device=device)
    g_sums = torch.zeros((2, num_splits, num_envs), dtype=torch.float, device=device)
    loss_cont_sums = torch.zeros((2, num_splits, num_envs), dtype=torch.float, device=device)
    loss_keep_cont_total = torch.tensor(0.0, device=device)

    for batch_index, data_env in enumerate(train_bar):
        # -----------------------
        # Micro-batch splitting
        # -----------------------
        pos_all_batch, indexs_batch = data_env[0], data_env[-1]
        mb_list = list(utils.microbatches(pos_all_batch, indexs_batch, gpu_batch_size))

        for j in range(2):  # split into halves
            for i in [i_ for i_ in range(len(mb_list)) if i_ % 2 == j]:
                batch_micro, indexs = mb_list[i] # indexs are the image indexes in dataset
                """
                Convert to loss-specific data, e.g.:
                    MoCo: out_q, out_k
                    SimSiam: z1, z2, p1, p2
                Also determines which of these require gradients
                """
                batch_micro = loss_module.pre_micro_batch(batch_micro, transform, normalize=True)
                # compute micro-batch loss
                loss_keep_cont, logits, labels = loss_module.compute_loss_micro(batch_micro)
                # compute g_i per split/env
                g_i = irm_calculator.gradients_for_half(logits, labels)
                # store g_i and loss sums in driver buffers
                # driver is responsible for placing into halves
                # update half_split_sz etc. (simplified placeholder)
                g_sums[j] += g_i.detach()
                loss_keep_cont_total += loss_keep_cont
                loss_module.post_micro_batch((None, None))  # if needed

        trained_samples += len(indexs_batch)

        # -----------------------
        # Macro-batch aggregation
        # -----------------------
        if (batch_index + 1) % gradients_accum_steps == 0:
            # compute IRM loss from g_sums
            irm_loss = irm_calculator.macro_batch_loss(g_sums, half_split_sz)
            # add to optimizer, divide by total number of samples if needed
            train_optimizer.zero_grad(set_to_none=True)
            # backward placeholder (real code would compute param grads using accumulated g_sums)
            # optimizer step
            train_optimizer.step()

            # -----------------------
            # Logging
            # -----------------------
            total_loss = (loss_keep_cont_total + irm_loss).item()
            desc_str = f'Train Epoch: [{epoch}/{epochs}] [{trained_samples}/{len(train_loader.dataset)}] ' \
                       f'Loss: {total_loss:.4f} LR: {train_optimizer.param_groups[0]["lr"]:.4f}'
            train_bar.set_description(desc_str)
            utils.write_log(desc_str, log_file=args.log_file)

            # -----------------------
            # Clear buffers for next macro-batch
            # -----------------------
            g_sums.zero_()
            loss_cont_sums.zero_()
            loss_keep_cont_total.zero_()
            half_split_sz.zero_()
            torch.cuda.empty_cache()

    return total_loss / trained_samples
