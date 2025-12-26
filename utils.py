from PIL import Image
#from torchvision import transforms
from torchvision.transforms import v2 as transforms
from torchvision.datasets import STL10, CIFAR10, CIFAR100, ImageFolder
import kornia.augmentation as K
#import cv2
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim, autograd
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils import data
import random
import os
import shutil

import pyvips

import torch
import re

import argparse
from types import SimpleNamespace
import sys
from collections import defaultdict

class ParseMixed(argparse.Action):
    def __init__(self, option_strings, dest, types=None, **kwargs):
        if types is None:
            raise ValueError("You must provide a list of types")
        self.types = types
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) != len(self.types):
            raise argparse.ArgumentError(
                self, f"Expected {len(self.types)} values, got {len(values)}"
            )
        converted = []
        for v, t in zip(values, self.types):
            if t is bool:
                v = str(v).lower()
                if v in ("true", "1", "yes", "y"):
                    converted.append(True)
                elif v in ("false", "0", "no", "n"):
                    converted.append(False)
                else:
                    raise argparse.ArgumentError(self, f"Invalid bool: {v}")
            else:
                converted.append(t(v))
        setattr(namespace, self.dest, converted)

def pretty_tensor_str(tensor, indent=0):
    """
    Pretty-print PyTorch tensor string with:
    - preserved truncation (ellipsis),
    - no device or grad info,
    - recursive indentation,
    - innermost lists (vectors) printed inline with brackets,
    - no extra brackets for 2D rows.

    Returns formatted string.
    """
    t = tensor.detach().cpu()
    s = str(t)

    # Remove device and grad_fn info
    s = re.sub(r", device='.*?'\)", ")", s)
    s = re.sub(r", grad_fn=<.*?>", "", s)

    # Remove tensor(...) wrapper
    if s.startswith("tensor(") and s.endswith(")"):
        s = s[len("tensor("):-1]

    s = s.strip()
    shape = t.shape

    def recursive_format(s, indent_level, level):
        # Trim outer brackets
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1].strip()

        if not s:
            return ' ' * indent_level + '[]'

        # For 1D tensor, just print inline
        if len(shape) == 1 or level == len(shape) - 1:
            return ' ' * indent_level + '[' + s + ']'

        # For 2D tensor at second-last level, print rows inline without extra brackets
        if len(shape) >= 2 and level == len(shape) - 2:
            # Split rows by commas outside brackets
            elems = []
            level_brackets = 0
            current = []
            for c in s:
                if c == '[':
                    level_brackets += 1
                elif c == ']':
                    level_brackets -= 1
                if c == ',' and level_brackets == 0:
                    elems.append(''.join(current).strip())
                    current = []
                else:
                    current.append(c)
            if current:
                elems.append(''.join(current).strip())

            lines = [' ' * indent_level + '[']
            for e in elems:
                lines.append(' ' * (indent_level + 2) + e)
            lines.append(' ' * indent_level + ']')
            return '\n'.join(lines)

        # Otherwise, split top-level elements and recurse
        elems = []
        level_brackets = 0
        current = []
        for c in s:
            if c == '[':
                level_brackets += 1
            elif c == ']':
                level_brackets -= 1
            if c == ',' and level_brackets == 0:
                elems.append(''.join(current).strip())
                current = []
            else:
                current.append(c)
        if current:
            elems.append(''.join(current).strip())

        lines = [' ' * indent_level + '[']
        for e in elems:
            lines.append(recursive_format(e, indent_level + 2, level + 1))
        lines.append(' ' * indent_level + ']')
        return '\n'.join(lines)

    return recursive_format(s, indent, 0)

def pyvips_loader(path):
    image = pyvips.Image.new_from_file(path, access="sequential")
    arr = np.ndarray(buffer=image.write_to_memory(),
                     shape=[image.height, image.width, image.bands],
                     dtype=np.uint8)
    tensor = torch.from_numpy(arr).permute(2,0,1).float() / 255.0  # CHW float
    return tensor
    #return Image.fromarray(arr)

class STL10Pair(STL10):
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return pos_1, pos_2, target


class STL10Pair_Index(STL10):
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return pos_1, pos_2, target, index


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

class CIFAR10Pair_Index(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target, index

class CIFAR100Pair(CIFAR100):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

class CIFAR100Pair_Index(CIFAR100):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target, index

def find_classes(directory, class_to_idx_fun):
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = [entry.name for entry in os.scandir(directory) if entry.is_dir()]
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: class_to_idx_fun(cls_name) for cls_name in classes}
    return classes, class_to_idx

class Imagenet_idx(ImageFolder):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None, class_to_idx=None):
        self.class_to_idx = class_to_idx
        super(Imagenet_idx, self).__init__(root, transform, target_transform, loader=pyvips_loader)
        self.index_pos = -1
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        
        if False and self.transform is not None:
            pos = self.transform(image)
        else:
            pos = image
        if False and self.target_transform is not None:
            target = self.target_transform(target)

        return pos, target, index

    def find_classes(self, directory):
        if self.class_to_idx:
            return find_classes(directory, self.class_to_idx)
        else:
           return super(Imagenet_idx, self).find_classes(directory) 

class Imagenet(ImageFolder):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None, class_to_idx=None):
        self.class_to_idx = class_to_idx
        super(Imagenet, self).__init__(root, transform, target_transform, loader=pyvips_loader)
        self.index_pos = None

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if False and self.transform is not None:
            pos = self.transform(image)
        else:
            pos = image
        if False and self.target_transform is not None:
            target = self.target_transform(target)

        return pos, target

    def find_classes(self, directory):
        if self.class_to_idx:
            return find_classes(directory, self.class_to_idx)
        else:
           return super(Imagenet, self).find_classes(directory) 

class Imagenet_idx_pair(ImageFolder):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None, class_to_idx=None):
        self.class_to_idx = class_to_idx
        super(Imagenet_idx_pair, self).__init__(root, transform, target_transform, loader=pyvips_loader)
        self.index_pos = -1

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if False and self.transform is not None:
            pos1 = self.transform(image)
            pos2 = self.transform(image)
        else:
            pos1 = image
            pos2 = image
        if False and self.target_transform is not None:
            target = self.target_transform(target)

        return pos1, pos2, target, index

    def find_classes(self, directory):
        if self.class_to_idx:
            return find_classes(directory, self.class_to_idx)
        else:
           return super(Imagenet_idx_pair, self).find_classes(directory) 

class Imagenet_pair(ImageFolder):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None, class_to_idx=None):
        self.class_to_idx = class_to_idx
        super(Imagenet_pair, self).__init__(root, transform, target_transform, loader=pyvips_loader)
        self.index_pos = None

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)

        if False and self.transform is not None:
            pos1 = self.transform(image)
            pos2 = self.transform(image)
        else:
            pos1 = image
            pos2 = image
        if False and self.target_transform is not None:
            target = self.target_transform(target)

        return pos1, pos2, target

    def find_classes(self, directory):
        if self.class_to_idx:
            return find_classes(directory, self.class_to_idx)
        else:
           return super(Imagenet_pair, self).find_classes(directory) 

class Imagenet_idx_pair_transformone(ImageFolder):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, root, transform_simple=None, transform_hard=None, target_transform=None, class_to_idx=None):
        self.class_to_idx = class_to_idx
        super(Imagenet_idx_pair_transformone, self).__init__(root, transform_simple, target_transform, loader=pyvips_loader)
        self.transform_hard = transform_hard
        self.index_pos = -1

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        image = self.loader(path)
        if False and self.transform is not None:
            pos1 = self.transform(image)
            pos2 = self.transform(image)
        else:
            pos1 = image
            pos2 = image
        if False and self.transform_hard is not None:
            pos1_hard = self.transform_hard(image)
            pos2_hard = self.transform_hard(image)
        else:
            pos1_hard = image
            pos2_hard = image
        if False and self.target_transform is not None:
            target = self.target_transform(target)

        return pos1, pos2, pos1_hard, pos2_hard, target, index

    def find_classes(self, directory):
        if self.class_to_idx:
            return find_classes(directory, self.class_to_idx)
        else:
           return super(Imagenet_idx_pair_transformone, self).find_classes(directory) 

class IndexDataset(Dataset):
    def __init__(self, size):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return idx
        
class MutableSampler(Sampler):
    def __init__(self, indices=None):
        self.indices = indices or []

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def set_indices(self, new_indices):
        self.indices = new_indices

class MutableBatchSampler:
    def __init__(self, batch_size, drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.indices = []

    def __iter__(self):
        for i in range(0, len(self.indices), self.batch_size):
            batch = self.indices[i:i + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue  # skip the last incomplete batch
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.indices) // self.batch_size
        else:
            return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def set_indices(self, new_indices):
        self.indices = new_indices

class LoaderManager:
    def __init__(self, dataset, num_passes, batched=True, **loader_kwargs):
        self.dataset = dataset
        self.num_passes = num_passes

        if num_passes > 1:
            # one sampler per pass
            if batched:
                self.samplers = [MutableBatchSampler(loader_kwargs['batch_size'], loader_kwargs['drop_last']) for _ in range(num_passes)]
                # create persistent loaders once
                loader_kwargs.pop('batch_size', None)
                loader_kwargs.pop('shuffle', None)
                loader_kwargs.pop('drop_last', None)
                self.loaders = [DataLoader(dataset, batch_sampler=s, **loader_kwargs) for s in self.samplers]
            else:
                self.samplers = [MutableSampler([]) for _ in range(num_passes)]
                # create persistent loaders once
                self.loaders = [DataLoader(dataset, sampler=s, **loader_kwargs) for s in self.samplers]

        else:           
            self.samplers = []
            self.loaders = [
                DataLoader(dataset, shuffle=True, **loader_kwargs)
            ]

    def new_macro_batch(self, indices):
        """Set a new shuffled order for all passes."""
        random.shuffle(indices)
        for s in self.samplers:
            s.set_indices(indices)

    def get_pass_iter(self, pass_idx):
        """Get an iterator over the loader for the given pass."""
        return iter(self.loaders[pass_idx])

    def shutdown(self):
        """Explicitly shut down persistent workers, iterators, and clear references."""
        # Kill workers for all loaders
        for dl in self.loaders:
            it = getattr(dl, "_iterator", None)
            if it is not None:
                it._shutdown_workers()  # terminate subprocesses

        # Drop any saved iterators if you decide to keep them in the future
        if hasattr(self, "iters"):
            self.iters = None

        # Clear references so GC can free memory
        self.loaders = []
        self.samplers = []
        self.dataset = None

def group_crossentropy(logits, labels, batchsize):
    sample_dim, label_dim = logits.size(0), logits.size(1)
    logits_exp = logits.exp()
    weights = torch.ones_like(logits_exp)
    weights[:, 1:] *= (batchsize-2)/(label_dim-1)
    softmax_loss = (weights * logits_exp) / (weights * logits_exp).sum(1).unsqueeze(1)
    cont_loss_env = torch.nn.NLLLoss()(torch.log(softmax_loss), labels)
    return cont_loss_env


def info_nce_loss(features, batch_size, temperature):
    # 'features' is a tensor of two views concatenated along dim=0
    # 'batch_size' is the length of the first view 

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0) # (2*batch_size,) of [0,batch_size)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() # (2*batch_size,2*batch_size) of labels{i,j]=True if labels[i]==labels[j]
    labels = labels.to(features.device)

    # features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T) # (2*batch_size,2*batch_size)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device) # (2*batch_size,2*batch_size) w/ True along the diagonal
    # When you do boolean indexing like labels[~mask], PyTorch (and NumPy) flattens the result into a 1D tensor of just the selected elements.
    # The order is row-major (C-order) in PyTorch (same as NumPy): pick a row, go across columns, move to next row.
    # each row corresponds to one anchor, and the columns are the other samples (diagonal removed).
    # PyTorch uses row-major storage. That means the 1D array FILLS the new 2D array row by row.
    labels = labels[~mask].view(labels.shape[0], -1) # (2*batch_size, 2*batch_size-1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # For each row i (an anchor), columns correspond to all other examples j != i in the concatenated 
    # batch (ordered by original column order, but with the diagonal element removed).
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    # each row has exactly one positive after removing the diagonal
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1) # (2*batch_size,1) of positive similarity scalars

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1) # (2B, 2B-2)

    logits = torch.cat([positives, negatives], dim=1) # (2B, 2B-1)
    # because positives are put in the first column, the correct class index (for the positive) is 0 for every row.
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    logits = logits / temperature
    return logits, labels


def info_nce_loss_update(features, batch_size, temperature):
    # 'features' is a tensor of two views concatenated along dim=0
    # 'batch_size' is the length of the first view 

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0) # (2*batch_size,) of [0,batch_size)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() # (2*batch_size,2*batch_size) of True where index match in both views
    labels = labels.to(features.device)
    # split_matrixs = torch.cat([split_matrix, split_matrix], dim=0).to(features.device)
    index_sequence = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0).to(features.device) # (2*batch_size,) of [0,batch_size)
    index_sequence = index_sequence.unsqueeze(0).expand(2*batch_size, 2*batch_size) # (2*batch_size,2*batch_size)

    # features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T) # (2*batch_size, 2*batch_size)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    # When you do boolean indexing like labels[~mask], PyTorch (and NumPy) flattens the result into a 1D tensor of just the selected elements.
    # The order is row-major (C-order) in PyTorch (same as NumPy): pick a row, go across columns, move to next row.
    # each row corresponds to one anchor, and the columns are the other samples (diagonal removed).
    # PyTorch uses row-major storage. That means the 1D array FILLS the new 2D array row by row.
    labels = labels[~mask].view(labels.shape[0], -1) # (2*batch_size, 2*batch_size-1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) # (2*batch_size, 2*batch_size-1)
    index_sequence = index_sequence[~mask].view(index_sequence.shape[0], -1) # (2*batch_size, 2*batch_size-1)
    # For each row i (an anchor), columns correspond to all other examples j != i in the concatenated 
    # batch (ordered by original column order, but with the diagonal element removed).

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1) # (2*batch_size, P)
    positive_index = index_sequence[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1) # (2*batch_size, N)
    negative_index = index_sequence[~labels.bool()].view(labels.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1) # (2*batch_size, D)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device) # (2*batch_size,)
    indexs = torch.cat([positive_index, negative_index], dim=1) # (2*batch_size, P+N)

    logits = logits / temperature
    return logits, labels, indexs

def moco_supcon_softenv_ce(
    z_q, z_all,
    y_q, y_all,
    w_all, # (N,) - denominator = positives & negatives
    tau,
    NEG=-1e9,
    supcon=True,
):
    """
    Returns L_env (scalar) using CrossEntropyLoss
        z_q: (B, D) query embeddings (anchors)
        z_all: (B, N=B+K) key embeddings (positives and negatives)
        y_q: (B,) class labels for queries
        y_all: (N=B+K,) labels for [z_k ; z_queue]
        w_all: (N=B+K,) soft env weights for keys
        temperature tau
    """

    device = z_q.device
    B, N = z_all.size()
    eps = 1e-12

    # --- similarities ---
    logits = (z_q @ z_all.T) / tau                  # (B, N)

    if supcon:
        # --- SupCon positive mask ---
        pos_mask = (y_q[:, None] == y_all[None, :])     # (B, N)
        pos_mask[:, :B].fill_diagonal_(False)
    else:
        # vanilla MoCo
        pos_mask = torch.zeros(N, dtype=torch.bool, device=device)
        pos_mask[:, :B].fill_diagonal_(True)

    # --- env gating (keys only) ---
    log_w = torch.log(w_all.clamp_min(eps))        # (N,)
    logits_env = logits + log_w[None, :]            # (B, N)

    # --- collapse positives -> single logit ---
    pos_logits = logits_env.masked_fill(~pos_mask, NEG)
    l_pos = torch.logsumexp(pos_logits, dim=1, keepdim=True)  # (B, 1)

    # --- negatives stay separate ---
    neg_logits = logits_env.masked_fill(pos_mask, NEG)       # (B, N)

    # --- CE-style logits ---
    ce_logits = torch.cat([l_pos, neg_logits], dim=1)        # (B, 1+N)
    labels = torch.zeros(B, dtype=torch.long, device=device)
    print()
    print(B,N)
    print(labels.size())
    return ce_logits, labels

def moco_loss_update(features, batch_size, weights, ssl_type, queue, dataset_idx, dataset, moco_temp, NEG=-1e9):
    # 'features' is a tensor of two views concatenated along dim=0
    # 'batch_size' is the length of the first view 
    # 'dataset_idx' are the samples' indices in the dataset
    # 'weights' are weights of a partition of ALL samples in the dataset
    device = features.device
    out_q = features[:batch_size]
    out_k = features[batch_size:]
    k_queue, idx_queue = queue.get(queue.queue_size, advance=False, idx=True) # 'idx_queue' are dataset indices of samples in queue
    k_all = torch.cat([out_k, k_queue], dim=0) # (N,D), N=B+K 
    k_indices_all = torch.cat([dataset_idx.to(device, non_blocking=True), idx_queue.to(device, non_blocking=True)], dim=0)
    w_all = weights[k_indices_all]

    def get_targets(idcs, dataset, device):
        targets = [dataset.targets[i] for i in idcs]
        if dataset.target_transform is not None:
            labels = [dataset.target_transform(t) for t in targets]
        else:
            labels = targets
        return torch.tensor(labels, dtype=torch.long, device=device)
    if ssl_type == 'moco':
        y_batch, y_all = None, None
    elif ssl_type == 'mocosupcon':
        y_batch = get_targets(dataset_idx, dataset, device)
        y_queue = get_targets(idx_queue, dataset, device)
        y_all = torch.cat([y_batch, y_queue], dim=0) # (N,)

    logits, labels = moco_supcon_softenv_ce(out_q, k_all, y_batch, y_all, w_all, moco_temp, NEG=NEG, supcon=ssl_type=='mocosupcon')
    return logits, labels

def penalty(logits, y, loss_function, mode='w', batchsize=None):
    assert((logits.size(0) % 2) == 0) 
    if mode == 'w':
        scale = torch.ones((1, logits.size(-1))).cuda(non_blocking=True).requires_grad_()
        try:
            loss1 = loss_function(logits[::2] * scale, y[::2])
            loss2 = loss_function(logits[1::2] * scale, y[1::2])
        except:
            assert batchsize is not None
            loss1 = loss_function(logits[::2] * scale, y[::2], batchsize)
            loss2 = loss_function(logits[1::2] * scale, y[1::2], batchsize)
        grad1 = autograd.grad(loss1, [scale], create_graph=True)[0]
        grad2 = autograd.grad(loss2, [scale], create_graph=True)[0]
    elif mode == 'f':
        pass
    return torch.sum(grad1*grad2)


class update_split_dataset(data.Dataset):
    def __init__(self, feature_bank1, feature_bank2):
        """Initialize and preprocess the Dsprite dataset."""
        self.feature_bank1 = feature_bank1
        self.feature_bank2 = feature_bank2

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        feature1 = self.feature_bank1[index]
        feature2 = self.feature_bank2[index]

        return feature1, feature2, index

    def __len__(self):
        """Return the number of images."""
        return self.feature_bank1.size(0)


# Update split online
def auto_split(net, update_loader, soft_split_all, temperature, irm_temp, loss_mode='v2', irm_mode='v1', irm_weight=10, constrain=False, cons_relax=False, nonorm=False, log_file=None):
    # irm mode: v1 is original irm; v2 is variance (not use)
    
    transform = update_loader.dataset.transform
    target_transform = update_loader.dataset.target_transform

    low_loss, constrain_loss = 1e5, torch.Tensor([0.])
    cnt, best_epoch, training_num = 0, 0, 0
    num_env = soft_split_all.size(1)

    # optimizer and schedule
    pre_optimizer = torch.optim.Adam([soft_split_all], lr=0.5, weight_decay=0.)
    pre_scheduler = MultiStepLR(pre_optimizer, [5, 35], gamma=0.2, last_epoch=-1)

    for epoch in range(100):
        risk_all_list, risk_cont_all_list, risk_penalty_all_list, risk_constrain_all_list, training_num = [],[],[],[], 0
        net.eval()
        for batch_idx, (pos_, target, idx) in enumerate(update_loader):
            training_num += len(pos_)
            with torch.no_grad():
                
                pos_ = pos_.cuda(non_blocking=True)

                if transform is not None:
                    pos_1 = transform(pos_)
                    pos_2 = transform(pos_)
                if target_transform is not None:
                    target = target_transform(target)
                
                _, feature_1 = net(pos_1)
                _, feature_2 = net(pos_2)

            loss_cont_list, loss_penalty_list = [], []

            """
            # Option 1. use probability directly
            soft_split = F.softmax(soft_split_all, dim=-1)
            for env_idx in range(num_env):
                loss_weight = torch.gather(soft_split[:, env_idx], dim=1, index=indexs)
                cont_loss_env_sample = (loss_weight*loss_original).sum(1)
                cont_loss_env = (cont_loss_env_sample * torch.cat([soft_split[:, env_idx], soft_split[:, env_idx]], dim=0)).sum(0)
                loss_cont_list.append(cont_loss_env)
                penalty_irm = torch.autograd.grad(cont_loss_env, [scale], create_graph=True)[0]
                loss_penalty_list.append(penalty_irm)
            risk_final = - (loss_cont_list.sum() + loss_penalty_list.sum())
            """

            # Option 2. use soft split
            param_split = F.softmax(soft_split_all[idx], dim=-1) # positive, normalized
            if irm_mode == 'v1': # original
                for env_idx in range(num_env):

                    logits, labels, indexs = info_nce_loss_update(torch.cat([feature_1, feature_2], dim=0), feature_1.size(0), temperature=1.0)

                    loss_weight = param_split[:, env_idx][indexs]
                    logits_cont = logits / temperature
                    # here we change the contrastive loss to the soft version to enable the sample weight
                    cont_loss_env = soft_contrastive_loss(logits_cont, labels, loss_weight, mode=loss_mode, nonorm=nonorm)

                    scale = torch.ones((1, logits.size(-1))).cuda(non_blocking=True).requires_grad_()
                    logits_pen = logits / irm_temp
                    cont_loss_env_scale1 = soft_contrastive_loss(logits_pen[::2]*scale, labels[::2], loss_weight[::2], mode=loss_mode, nonorm=nonorm)
                    cont_loss_env_scale2 = soft_contrastive_loss(logits_pen[1::2]*scale, labels[1::2], loss_weight[1::2], mode=loss_mode, nonorm=nonorm)
                    penalty_irm1 = torch.autograd.grad(cont_loss_env_scale1, [scale], create_graph=True)[0]
                    penalty_irm2 = torch.autograd.grad(cont_loss_env_scale2, [scale], create_graph=True)[0]
                    loss_cont_list.append((cont_loss_env1 + cont_loss_env2)/2)
                    loss_penalty_list.append(torch.sum(penalty_irm1*penalty_irm2))

                cont_loss_epoch = torch.stack(loss_cont_list).mean()
                inv_loss_epoch = torch.stack(loss_penalty_list).mean()
                risk_final = - (cont_loss_epoch + irm_weight*inv_loss_epoch)


            elif irm_mode == 'v2': # variance (not use)
                for env_idx in range(num_env):
                    logits, labels, indexs = info_nce_loss_update(torch.cat([feature_1, feature_2], dim=0), feature_1.size(0), temperature=1.0)
                    loss_weight = param_split[:, env_idx][indexs]
                    logits_cont = logits / temperature
                    cont_loss_env = soft_contrastive_loss(logits_cont, labels, loss_weight, mode=loss_mode, nonorm=nonorm)
                    loss_cont_list.append(cont_loss_env)

                inv_loss_epoch = torch.var(torch.stack(loss_cont_list))
                cont_loss_epoch = torch.stack(loss_cont_list).mean()
                risk_final = - (cont_loss_epoch + irm_weight*inv_loss_epoch)

            if constrain > 0.: # constrain to avoid the imbalance problem
                if nonorm:
                    """
                    Each example should have confident (low-entropy) predictions, but across the batch they should be evenly distributed across classes.
                    """
                    constrain_loss = 0.2*(- cal_entropy(param_split.mean(0), dim=0) + cal_entropy(param_split, dim=1).mean())
                else:
                    if cons_relax: # relax constrain to make item num of groups no more than 2:1
                        """
                        Don't let the model's global prediction distribution get too peaky - stay at least roughly balanced (entropy >= 0.6365). 
                        If it starts collapsing, push it back.
                        For 2 classes, H([0.55,0.45])~0.688, H([0.7,0.3])~0.611. So 0.6365 corresponds roughly to a 65-35 class split.  
                        So, for 99-1 split, entropy is ~ 0, so relu() gives ~ 0.6365, which increases total loss.
                        For 50-50 split, entropy is ~ 0.69, so relu() gives 0, i.e., loss isn't increased.
                        Net result is that splits better than 2:1 are capped and don't decrease the loss anymore.
                        """
                        constrain_loss = torch.relu(0.6365 - cal_entropy(param_split.mean(0), dim=0))
                    else:
                        """
                        Rewards diversity across the batch - it is more positive when the model collapses to one class. 
                        Here, entropy always > 0. It's bigger when there's MORE diversity (more even split), and its
                        negation is SMALLER, i.e. loss goes DOWN.
                        """
                        constrain_loss = - cal_entropy(param_split.mean(0), dim=0)#  + cal_entropy(param_split, dim=1).mean()
                constrain_loss *= constrain
                risk_final += constrain_loss


            pre_optimizer.zero_grad()
            risk_final.backward()
            pre_optimizer.step()

            risk_all_list.append(risk_final.item())
            risk_cont_all_list.append(-cont_loss_epoch.item())
            risk_penalty_all_list.append(-inv_loss_epoch.item())
            risk_constrain_all_list.append(constrain_loss.item())
            soft_split_print = soft_split_all[:1].clone().detach()
            if epoch > 0:
                print('\rUpdating Env [%d/%d] [%d/%d]  Loss: %.2f  Cont_Risk: %.2f  Inv_Risk: %.2f  Cons_Risk: %.2f  Cnt: %d  Lr: %.4f  Inv_Mode: %s  Soft Split: %s'
                      %(epoch, 100, training_num, len(update_loader.dataset), sum(risk_all_list)/len(risk_all_list), sum(risk_cont_all_list)/len(risk_cont_all_list), sum(risk_penalty_all_list)/len(risk_penalty_all_list),
                        sum(risk_constrain_all_list)/len(risk_constrain_all_list), cnt, pre_optimizer.param_groups[0]['lr'], irm_mode, 
                        F.softmax(soft_split_print, dim=-1).tolist()), end='', flush=True)


        pre_scheduler.step()
        avg_risk = sum(risk_all_list)/len(risk_all_list)
        avg_cont_risk = sum(risk_cont_all_list)/len(risk_cont_all_list)
        avg_inv_risk = sum(risk_penalty_all_list)/len(risk_penalty_all_list)

        if epoch == 0:
            write_log("Initial Risk: %.2f  Cont_Risk: %.2f  Inv_Risk: %.2f" %(avg_risk, avg_cont_risk, avg_inv_risk), log_file=log_file, print_=True)
            soft_split_best = soft_split_all.clone().detach()
        if avg_risk < low_loss:
            low_loss = avg_risk
            soft_split_best = soft_split_all.clone().detach()
            best_epoch = epoch
            cnt = 0
        else:
            cnt += 1

        if epoch > 50 and cnt >= 5 or epoch == 60:
            write_log('\nLoss not down. Break down training.  Epoch: %d  Loss: %.2f' %(best_epoch, low_loss), log_file=log_file, print_=True)
            write_log('Updating Env [%d/%d] [%d/%d]  Loss: %.2f  Cont_Risk: %.2f  Inv_Risk: %.2f  Cons_Risk: %.2f  Cnt: %d  Lr: %.4f  Inv_Mode: %s'
                      %(epoch, 100, training_num, len(update_loader.dataset), sum(risk_all_list)/len(risk_all_list), sum(risk_cont_all_list)/len(risk_cont_all_list), sum(risk_penalty_all_list)/len(risk_penalty_all_list),
                        sum(risk_constrain_all_list)/len(risk_constrain_all_list), cnt, pre_optimizer.param_groups[0]['lr'], irm_mode), log_file=log_file)
            final_split_softmax = F.softmax(soft_split_best, dim=-1)
            write_log('%s' %(pretty_tensor_str(final_split_softmax)), log_file=log_file, print_=True)
            group_assign = final_split_softmax.argmax(dim=1)           
            num_groups = final_split_softmax.size(1)  # number of groups
            counts = torch.bincount(group_assign, minlength=num_groups)
            group_str = [f"group{i+1} {counts[i]}" for i in range(num_groups)]
            write_log('Debug: ' + ' '.join(group_str), log_file=log_file, print_=True)
            return soft_split_best


# update split offline
# out_1, out_2 are already post transform() and are in cpu
def auto_split_offline(out_1, out_2, soft_split_all, temperature, irm_temp, loss_mode='v2', irm_mode='v1', irm_weight=10, constrain=False, 
            cons_relax=False, nonorm=False, log_file=None, batch_size=3096, num_workers=4, prefetch_factor=2, persistent_workers=True,
            ssl_type='simclr', queue=None, dataset=None):
    # 'out_1', 'out_2' are features of samples in the whole dataset, their order corresponds to the dataset original order
    # 'irm_mode': 'v1' is original irm; 'v2' is variance
    low_loss, constrain_loss = 1e5, torch.Tensor([0.])
    cnt, best_epoch, training_num = 0, 0, 0
    num_env = soft_split_all.size(1)
    # optimizer and schedule
    pre_optimizer = torch.optim.Adam([soft_split_all], lr=0.5, weight_decay=0.)
    pre_scheduler = MultiStepLR(pre_optimizer, [5, 35], gamma=0.2, last_epoch=-1)

    # dataset and dataloader
    traindataset = update_split_dataset(out_1, out_2)
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
        prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, pin_memory=True, drop_last=False)

    for epoch in range(100):
        risk_all_list, risk_cont_all_list, risk_penalty_all_list, risk_constrain_all_list, training_num = [],[],[],[], 0

        for feature_1, feature_2, idx in trainloader: # 'idx' is the index in the dataset
            feature_1, feature_2 = feature_1.cuda(non_blocking=True), feature_2.cuda(non_blocking=True)
            loss_cont_list, loss_penalty_list = [], [] # per-env aggregators
            training_num += len(feature_1)

            param_split = F.softmax(soft_split_all, dim=-1) # positive, normalized across domains
            for env_idx in range(num_env):
                if ssl_type == 'simclr':
                    # indexs[i, j] = original image ID generating the j-th logit for anchor i
                    logits, labels, indexs = info_nce_loss_update(torch.cat([feature_1, feature_2], dim=0), feature_1.size(0), temperature=1.0)
                    # For anchor i and comparison j, attach the environment-specific weight of the compared image.
                    # indexs[i, j] tells you which original image produced the j-th logit for anchor i, 
                    # and param_split[:, env_idx][indexs] replaces each image ID with its environment-specific weight.
                    # loss_weight[i, j] = param_split[indexs[i, j], env_idx], loss_weight.shape == indexs.shape
                    loss_weight = param_split[idx, env_idx][indexs] 
                    logits_cont = logits / temperature
                    """
                    They want a contrastive loss where:
                        each anchor has 1 positive + many negatives (InfoNCE / SimCLR style)
                        negatives are reweighted by environment-dependent weights
                        positives may also be reweighted
                        normalization can be per-sample or global
                    logits:  (N, K)   # N = 2B anchors, K = 1 + (#negatives)
                    labels:  (N,)     # always 0 (positive is at index 0)
                    weights: (N, K)   # per-(anchor, comparison) weights
                    """
                    cont_loss_env = soft_contrastive_loss(logits_cont, labels, loss_weight, mode=loss_mode, nonorm=nonorm)

                    if irm_mode == 'v1': # original
                        scale = torch.ones((1, logits.size(-1))).cuda(non_blocking=True).requires_grad_()
                        logits_pen = logits / irm_temp
                        cont_loss_env_scale1 = soft_contrastive_loss(logits_pen[::2]*scale, labels[::2], loss_weight[::2], mode=loss_mode, nonorm=nonorm)
                        cont_loss_env_scale2 = soft_contrastive_loss(logits_pen[1::2]*scale, labels[1::2], loss_weight[1::2], mode=loss_mode, nonorm=nonorm)

                elif ssl_type == 'moco' or ssl_type == 'mocosupcon':
                    weights_all = param_split[:, env_idx]
                    NEG = -1e9
                    logits, labels = moco_loss_update(torch.cat([feature_1, feature_2], dim=0), feature_1.size(0), weights_all, ssl_type, 
                                        queue=queue, dataset_idx=idx, dataset=dataset, moco_temp=temperature, NEG=NEG)
                    # sum over batch, per env handled by driver
                    # get the samples that have POSITIVES (column 0)
                    l_pos = logits[:, 0]
                    valid = l_pos > NEG
                    print()
                    print(valid)
                    print(vlaid.size())
                    print(logits.size())
                    print(labels.size())
                    logits = logits[valid]
                    labels = labels[valid]
                    w_anchors = weights_all[idx]
                    loss_anchors = F.cross_entropy(logits, labels, reduction='none')
                    eps = 1e-12
                    # --- anchor gating ---
                    cont_loss_env = (w_anchors * loss_anchors).sum()                      
                    if nonorm:
                        sample_dim = logits.size(0)
                        cont_loss_env /= sample_dim
                    else:
                        cont_loss_env /= w_anchors.sum().clamp_min(eps) 

                    if irm_mode == 'v1': # original
                        scale = torch.ones((1, logits.size(-1))).cuda(non_blocking=True).requires_grad_()
                        logits_pen = logits / irm_temp

                        loss_per_anchor = F.cross_entropy(scale*logits[::2], labels[::2], reduction='none')
                        cont_loss_env_scale1 = (loss_per_anchor * weights[::2]).sum() / weights[::2].sum()
                        loss_per_anchor = F.cross_entropy(scale*logits[1::2], labels[1::2], reduction='none')
                        cont_loss_env_scale2 = (loss_per_anchor * weights[1::2]).sum() / weights[1::2].sum()
                # end if ssl_type == ...

                # calculate IRM penalty
                penalty_irm1 = torch.autograd.grad(cont_loss_env_scale1, [scale], create_graph=True)[0]
                penalty_irm2 = torch.autograd.grad(cont_loss_env_scale2, [scale], create_graph=True)[0]

                loss_cont_list.append(cont_loss_env)
                loss_penalty_list.append(torch.sum(penalty_irm1*penalty_irm2))
            # end for env_idx in range(num_env):

            if irm_mode == 'v1': # IRM
                inv_loss_epoch = torch.stack(loss_penalty_list).mean()
            elif irm_mode == 'v2': # VREx (not used in the paper)
                inv_loss_epoch = torch.var(torch.stack(loss_cont_list))
            cont_loss_epoch = torch.stack(loss_cont_list).mean()
            risk_final = - (cont_loss_epoch + irm_weight*inv_loss_epoch)

            if constrain > 0: # constrain to avoid the imbalance problem
                if nonorm:
                    constrain_loss = 0.2*(- cal_entropy(param_split.mean(0), dim=0) + cal_entropy(param_split, dim=1).mean())
                else:
                    if cons_relax: # relax constrain to make item num of groups no more than 2:1
                        constrain_loss = torch.relu(0.6365 - cal_entropy(param_split.mean(0), dim=0))
                    else:
                        constrain_loss = - cal_entropy(param_split.mean(0), dim=0)#  + cal_entropy(param_split, dim=1).mean()
                constrain_loss = constrain * constrain_loss
                risk_final += constrain_loss

            pre_optimizer.zero_grad()
            risk_final.backward()
            pre_optimizer.step()

            risk_all_list.append(risk_final.item())
            risk_cont_all_list.append(-cont_loss_epoch.item())
            risk_penalty_all_list.append(-inv_loss_epoch.item())
            risk_constrain_all_list.append(constrain_loss.item())
            soft_split_print = soft_split_all[:1].clone().detach()
            if epoch > 0:
                print('\rUpdating Env [%d/%d] [%d/%d] Loss: %.2f Cont_Risk: %.2f Inv_Risk: %.2f Cons_Risk: %.2f Cnt: %d Lr: %.4f Inv_Mode: %s Soft Split: [%s]'
                      %(epoch, 100, training_num, len(trainloader.dataset), sum(risk_all_list)/len(risk_all_list), sum(risk_cont_all_list)/len(risk_cont_all_list), 
                        sum(risk_penalty_all_list)/len(risk_penalty_all_list),
                        sum(risk_constrain_all_list)/len(risk_constrain_all_list), cnt, pre_optimizer.param_groups[0]['lr'], irm_mode, 
                        ", ".join("%.4f" % v for v in F.softmax(soft_split_print, dim=-1)[0].tolist()),
                       ), end='', flush=True)

        pre_scheduler.step()
        avg_risk = sum(risk_all_list)/len(risk_all_list)
        avg_cont_risk = sum(risk_cont_all_list)/len(risk_cont_all_list)
        avg_inv_risk = sum(risk_penalty_all_list)/len(risk_penalty_all_list)

        if epoch == 0:
            write_log("Initial Risk: %.2f  Cont_Risk: %.2f  Inv_Risk: %.2f" % (avg_risk, avg_cont_risk, avg_inv_risk), log_file=log_file, print_=True)
            soft_split_best = soft_split_all.clone().detach()
        if avg_risk < low_loss:
            low_loss = avg_risk
            soft_split_best = soft_split_all.clone().detach()
            best_epoch = epoch
            cnt = 0
        else:
            cnt += 1

        if epoch > 50 and cnt >= 5 or epoch == 60:
            write_log('\nLoss not down. Stop training. Epoch: %d  Loss: %.2f' %(best_epoch, low_loss), log_file=log_file, print_=True)
            write_log('Updating Env [%d/%d] [%d/%d]  Loss: %.2f  Cont_Risk: %.2f  Inv_Risk: %.2f  Cons_Risk: %.2f  Cnt: %d  Lr: %.4f  Inv_Mode: %s'
                      %(epoch, 100, training_num, len(trainloader.dataset), sum(risk_all_list)/len(risk_all_list), sum(risk_cont_all_list)/len(risk_cont_all_list), sum(risk_penalty_all_list)/len(risk_penalty_all_list),
                        sum(risk_constrain_all_list)/len(risk_constrain_all_list), cnt, pre_optimizer.param_groups[0]['lr'], irm_mode), log_file=log_file)
            final_split_softmax = F.softmax(soft_split_best, dim=-1)
            write_log('%s' %(pretty_tensor_str(final_split_softmax)), log_file=log_file, print_=True)
            group_assign = final_split_softmax.argmax(dim=1)
            write_log('Debug:  group1 %d  group2 %d' %(group_assign.sum(), group_assign.size(0)-group_assign.sum()), log_file=log_file, print_=True)
            del trainloader
            return soft_split_best


# soft version of the contrastive loss
def soft_contrastive_loss(logits, labels, weights, mode='v1', nonorm=False):
    """
    This loss implements InfoNCE where negatives are reweighted inside the softmax according to environment-specific importance, 
    while positives remain fixed, and normalization controls gradient scale.    
    """
    if mode == 'v1':
        logits *= weights
        cont_loss_env = torch.nn.CrossEntropyLoss()(logits, labels)
    elif mode == 'v2':
        sample_dim, label_dim = logits.size(0), logits.size(1)
        logits_exp = logits.exp()
        # weight_pos: (N, 1), weight of the positive for anchor i
        # weight_neg: (N, K-1), weight of j-th negative for each anchor
        weight_pos, weight_neg = torch.split(weights, [1, label_dim-1], dim=1)
        """
        Normalize negative weights
        Per anchor: sum_j weight_neg_norm[i, j] = (K-1)
            average negative weight = 1
            negatives are reweighted relative to each other
            total negative mass is preserved  
        """
        weight_neg_norm = weight_neg / weight_neg.sum(1).unsqueeze(1) * (label_dim-1)
        """
        Construct final weights
            positive weight is set to 1       
            only negatives are reweighted
            All positives are equally important; environments affect how hard negatives are
            This is a design choice.
        """
        weights_new = torch.cat([torch.ones_like(weight_pos), weight_neg_norm], dim=1)
        softmax_loss = (weights_new*logits_exp) / (weights_new*logits_exp).sum(1).unsqueeze(1)
        cont_loss_env = torch.nn.NLLLoss(reduction='none')(torch.log(softmax_loss), labels)
        if nonorm:
            cont_loss_env = (cont_loss_env * weight_pos.squeeze()).sum() / sample_dim
        else:
            cont_loss_env = (cont_loss_env * weight_pos.squeeze()).sum() / weight_pos.sum()    # norm version

    return cont_loss_env


def assign_samples(data, split, env_idx):
    images_pos1, images_pos2, labels, idxs = data
    group_assign = split[idxs].argmax(dim=1)
    select_idx = torch.where(group_assign==env_idx)[0]
    return images_pos1[select_idx], images_pos2[select_idx]

def assign_features(feature1, feature2, idxs, split, env_idx):
    # Returns the indices of the maximum value of all elements in the input tensor.
    # There're 'env_num' groups in a split, so this returns the group [0,env_num) with the biggest value
    # i.e. which group the sample is asigned to
    # 'split' is a (dataset_size,env_num) of weights of sample i belonging to env j
    # feature1/2 are the corresponding features in a batch
    # idxs are their indices in the dataset
    group_assign = split[idxs].argmax(dim=1)
    # torch.where(condition) is identical to torch.nonzero(condition, as_tuple=True)
    # Returns a tuple of 1-D tensors, one for each dimension in input, each containing the indices 
    # (in that dimension) of all non-zero elements of input.
    # If input has n dimensions, then the resulting tuple contains n tensors of size z, 
    # where z is the total number of non-zero elements in the input tensor.
    # Select those samples that belong are in 'env' and have the largest value in 'split'
    select_idx = torch.where(group_assign==env_idx)[0]
    return feature1[select_idx], feature2[select_idx]

def assign_feature(feature, idxs, split, env_idx):
    # Returns the indices of the maximum value of all elements in the input tensor.
    # There're 'env_num' groups in a split, so this returns the group [0,env_num) with the biggest value
    # i.e. which group the sample is asigned to
    # 'split' is a (dataset_size,env_num) of weights of sample i belonging to env j
    # feature1/2 are the corresponding features in a batch
    # idxs are their indices in the dataset
    group_assign = split[idxs].argmax(dim=1)
    # torch.where(condition) is identical to torch.nonzero(condition, as_tuple=True)
    # Returns a tuple of 1-D tensors, one for each dimension in input, each containing the indices 
    # (in that dimension) of all non-zero elements of input.
    # If input has n dimensions, then the resulting tuple contains n tensors of size z, 
    # where z is the total number of non-zero elements in the input tensor.
    # Select those samples that belong are in 'env' and have the largest value in 'split'
    select_idx = torch.where(group_assign==env_idx)[0]
    return feature[select_idx]


def assign_idxs(idxs, split, env_idx):
    group_assign = split[idxs].argmax(dim=1)
    select_idx = torch.where(group_assign==env_idx)[0]
    return select_idx


def cal_entropy(prob, dim=1):
    return -(prob * prob.log()).sum(dim=dim)


def irm_scale(irm_loss, default_scale=-100):
    with torch.no_grad():
        scale =  default_scale / irm_loss.clone().detach()
    return scale

# SEED
def set_seed(seed):
    if_cuda = torch.cuda.is_available()
    torch.manual_seed(seed)
    if if_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_log(print_str, log_file, print_=False):
    if print_:
        print(print_str)
    if log_file is None:
        return
    with open(log_file, 'a') as f:
        f.write('\n')
        f.write(print_str)


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, mmin=0.1, mmax=2.0):
        self.min = mmin
        self.max = mmax
        # Ensure kernel size is odd and >= 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = max(1, kernel_size)
    
    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

# just follow the previous work -- DCL, NeurIPS2020

def make_train_transform(image_size=64, randgray=True, normalize='CIFAR', gpu=True):
    kernel_size = int(0.1 * image_size)
    if (kernel_size % 2) == 0:
        kernel_size += 1
        
    if (normalize == 'CIFAR') or (normalize == 'STL'):
        norm_mean = [0.4914, 0.4822, 0.4465]
        norm_std = [0.2023, 0.1994, 0.2010]
    elif normalize == 'ImageNet':
        norm_mean=[0.485, 0.456, 0.406]
        norm_std=[0.229, 0.224, 0.225]

    cpu_transform = transforms.Compose([
            #transforms.ToTensor(),  # <-- important: switch to tensor here
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)), # ratio=(0.75, 1.3333333333333333)
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2) if randgray else transforms.Lambda(lambda x: x),
            transforms.GaussianBlur(kernel_size=kernel_size),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])

    gpu_transform = K.AugmentationSequential(
        K.RandomResizedCrop((image_size, image_size), scale=(0.7,1.0)),
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJitter(0.4,0.4,0.4,0.1),
        K.RandomGrayscale(p=0.2) if randgray else nn.Identity(),
        K.RandomGaussianBlur((kernel_size,kernel_size), sigma=(0.1,2.0)),
        K.Normalize(mean=norm_mean, std=norm_std)
    )

    if gpu:
        return gpu_transform
    else:
        return cpu_transform
        
def make_test_transform(normalize='CIFAR'):
    if (normalize == 'CIFAR') or (normalize == 'STL'):
        norm_mean = [0.4914, 0.4822, 0.4465]
        norm_std = [0.2023, 0.1994, 0.2010]
    elif normalize == 'ImageNet':
        norm_mean=[0.485, 0.456, 0.406]
        norm_std=[0.229, 0.224, 0.225]

    cpu_transform = transforms.Compose([
        #transforms.ToTensor(),  # <-- important: switch to tensor here
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=norm_mean, std=norm_std)])

    gpu_transform = K.AugmentationSequential(
        K.Normalize(mean=norm_mean, std=norm_std)
    )

def atomic_save(state, is_best, filename='checkpoint.pth.tar', sync=True):
    filename_tmp = filename + ".tmp"
    torch.save(state, filename_tmp)

    try:
        # kaggle sometimes silently fails to replace the file. remove it to make sure it's gone
        if os.path.exists(filename):
            os.remove(filename)
        os.replace(filename_tmp, filename)
    except Exception as e:
        import sys, traceback
        print(f"[SAVE ERROR] Failed replacing {filename}: {e}", file=sys.stderr)
        traceback.print_exc()
        raise

    if is_best:
        dir_path = os.path.dirname(filename)
        best_filename = '{}/model_best.pth.tar'.format(dir_path)
        best_filename_tmp = filename + ".tmp"
        shutil.copyfile(filename, best_filename_tmp)
        try:
            # kaggle sometimes silently fails to replace the file. remove it to make sure it's gone
            if os.path.exists(best_filename):
                os.remove(best_filename)
            os.replace(best_filename_tmp, best_filename)
        except Exception as e:
            import sys, traceback
            print(f"[SAVE ERROR] Failed replacing {best_filename}: {e}", file=sys.stderr)
            traceback.print_exc()
            raise

    if sync:
        # Sync file data
        for p in [filename, best_filename] if is_best else [filename]:
            fd = os.open(p, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)

        # Sync the directory once (covers both files)
        dir_fd = os.open(os.path.dirname(filename) or ".", os.O_RDONLY)
        os.fsync(dir_fd)
        os.close(dir_fd)
   
def increasing_weight(pars, penalty_target, penalty_iters, epoch, epochs):
    penalty_warmup, scale, speed, eps, debug = pars
    if epoch < penalty_iters:
        return penalty_warmup
    # Exponential growth starting at small epsilon to avoid zero multiplication
    power = (epoch - penalty_iters) * speed
    w = penalty_warmup + eps * (scale ** power)
    penalty_weight = min(w, penalty_target)
    if debug > 0:
        print()
        print(f"penalty_warmup {penalty_warmup}, penalty_target {penalty_target}, penalty_iters {penalty_iters}, epoch {epoch}," +
              f" epochs {epochs}, speed {speed:.2f}, power {power:.4f}, w {w:.4f}, penalty_weight {penalty_weight:.4f}")
    return penalty_weight

class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True, active=False):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction
        self.active = active

    def update(self, dict_data, orig_shape=False):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self.active:
                if self._updates == 0 or (name not in self.ema_data):
                    previous_data = data
                else:
                    previous_data = self.ema_data[name]

                ema_data = self.ema * previous_data + (1 - self.ema) * data
                if self._oneminusema_correction:
                    # correction by 1/(1 - self.ema)
                    # so that the gradients amplitude backpropagated in data is independent of self.ema
                    ema_dict_data[name] = ema_data / (1 - self.ema)
                else:
                    ema_dict_data[name] = ema_data
                self.ema_data[name] = ema_data.clone().detach()
            else:
                 ema_dict_data[name] = torch.ones_like(data)
        if self.active:
            self._updates += 1

        if orig_shape:
            ema_dict_data = {k: v.view(dict_data[k].shape) for k,v in ema_dict_data.items()}
        return ema_dict_data

    def set_active(self, active):
        self.active = active

class NonExclusiveParser:
    """
    Explicit sentinel-based multi-subparser wrapper.

    Syntax:
        prog.py [base args] [-- name <args>] [-- name <args>] ...

    Everything before the first '--' is parsed by base_parser.
    Each '-- <name> ...' starts a subparser section until the next '--' or EOF.
    Allows multiple occurrences of the same subparser.
    """

    def __init__(self, *args, **kwargs):
        self.base_parser = argparse.ArgumentParser(*args, **kwargs)
        self.subparsers = {}

    def add_argument(self, *args, **kwargs):
        return self.base_parser.add_argument(*args, **kwargs)

    def add_subparser(self, name, **kwargs):
        sp = argparse.ArgumentParser(prog=name, add_help=False, **kwargs)
        self.subparsers[name] = sp
        return sp

    def parse_args(self, argv=None):
        argv = sys.argv[1:] if argv is None else list(argv)

        # Split into sections separated by '--'
        chunks, current = [], []
        for token in argv:
            if token == "--":
                if current:
                    chunks.append(current)
                    current = []
            else:
                current.append(token)
        if current:
            chunks.append(current)

        results = defaultdict(list)

        # Base parser: first chunk
        if chunks:
            base_chunk = chunks.pop(0)
            base_ns, unknown = self.base_parser.parse_known_args(base_chunk)
            if unknown:
                raise SystemExit(f"unrecognized base args: {unknown}")
            base_dict = vars(base_ns)
        else:
            base_dict = {}

        # Subparser sections
        for chunk in chunks:
            if not chunk:
                continue
            name, *args = chunk
            if name not in self.subparsers:
                raise SystemExit(f"unknown subparser '{name}'")
            sp = self.subparsers[name]
            ns = sp.parse_args(args)
            results[name].append(ns)

        # Merge base args at top level, and subparser results under their names
        final = dict(base_dict)
        for name, v in results.items():
            if len(v) == 1:
                final[name] = v[0]
            else:
                final[name] = v

        return argparse.Namespace(**final)

    """
    if __name__ == "__main__":
        parser = NonExclusiveParser(description="Sentinel-based non-exclusive subparsers")

        parser.add_argument("--foo", type=int)

        pa = parser.add_subparser("a")
        pa.add_argument("--x")
        pa.add_argument("--y", nargs="*")

        pb = parser.add_subparser("b")
        pb.add_argument("--v", type=int)

        args = parser.parse_args("--foo 10 -- a --x 1 -- b --v 2 -- a --x 3 --y 4 5".split())
        print(args)

        Namespace(
          _base=Namespace(foo=10),
          a=[Namespace(x='1', y=[]), Namespace(x='3', y=['4', '5'])],
          b=Namespace(v=2)
        )
    """    

def reset_optimizer(optimizer):
    """
    Reinitialize all dynamic state (e.g. momentum, exp_avg, exp_avg_sq, step)
    while preserving:
        - parameter groups
        - learning rates
        - momentum/betas
        - weight decay
        - references to model parameters
    Works for Adam, SGD, and similar torch.optim optimizers.
    """
    # Loop over parameter groups and parameters
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p, None)
            if state is not None:
                state.clear()  # remove old buffers

            # Rebuild empty state if optimizer would normally track it
            if isinstance(optimizer, torch.optim.SGD):
                # SGD with momentum keeps a momentum_buffer
                if group.get('momentum', 0) != 0:
                    optimizer.state[p] = {'momentum_buffer': torch.zeros_like(p.data)}
            elif isinstance(optimizer, torch.optim.Adam) or isinstance(optimizer, torch.optim.AdamW):
                optimizer.state[p] = {
                    'step': torch.tensor(0.0, dtype=torch.float32),
                    'exp_avg': torch.zeros_like(p.data),
                    'exp_avg_sq': torch.zeros_like(p.data)
                }
            else:
                # Generic fallback: just clear whatever state exists
                optimizer.state[p] = {}

