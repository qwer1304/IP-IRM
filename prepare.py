# Each descriptor describes a SINGLE combined dataset from the environments of a dataset
# These could be:
#   - training samples split out off the environment datasets
#   - validation samples split out of the environment datasets
#   - all samples from some environment(s) to be used for validation or testing
#   - the samples from different domains can be combined into a single dataset using ConcatenatedDomain class
#   - the resulting dataset can be wrapped into a dataset s.t. the target_transform can be changed dynamically\
"""
#   - all descriptors will use the same split for the same domains
descriptior:
{'dataset': dataset_class,
 'transform': transform_proc.
 'target_transform': target_transform_proc,
 'class_to_index': class_to_index_proc,
 'split': True/False, # doesn't apply to test envs
 'concat': True/False, # concatenate envs (in and out separately)
 'wrap': True/False, # for changeable target transform
 'target_pos': target_index,
 'excluded_envs': list of envs,
 'required_split': "in"/"out",
}
"""
import os
import torch
import argparse
import utils
import cv2
import numpy as np
import bisect
from torchvision.datasets import STL10, CIFAR10, CIFAR100, ImageFolder


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0, in_idx=None, out_idx=None):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    assert ((in_idx is not None) and (out_idx is not None)) or ((in_idx is None) and (out_idx is None))
    assert (in_idx is None) or (n == len(in_idx))
    if in_idx is None:
        keys = list(range(len(dataset)))
        np.random.RandomState(seed).shuffle(keys)
        keys_1 = keys[:n]
        keys_2 = keys[n:]
    else:
        keys_1 = in_idx
        keys_2 = out_idx
    if n == len(dataset):
        return dataset, keys_1, None, keys_2
    elif n == 0:
        return None, keys_1, dataset, keys_2
    else:
        return _SplitDataset(dataset, keys_1), keys_1, _SplitDataset(dataset, keys_2), keys_2

class ConcatDataset(torch.utils.data.Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

        # stitch targets if available
        self.targets = [t for d in self.datasets if hasattr(d, "targets") for t in d.targets]

        # optionally stitch other ImageFolder attributes
        if all(hasattr(d, "classes") for d in self.datasets):
            # keep global classes consistent
            self.classes = self.datasets[0].classes
            self.class_to_idx = self.datasets[0].class_to_idx
        if all(hasattr(d, "transform") for d in self.datasets):
            # keep global classes consistent
            self.transform = self.datasets[0].transform
        if all(hasattr(d, "target_tranform") for d in self.datasets):
            # keep global classes consistent
            self.target_transform = self.datasets[0].target_transform

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

class TargetTransformWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, target_transform=None, target_pos=None):
        self.dataset = dataset
        # stitch targets if available
        self.targets = self.dataset.targets if hasattr(self.dataset, "targets") else []

        # optionally stitch other ImageFolder attributes
        if hasattr(self.dataset, "classes"):
            # keep global classes consistent
            self.classes = self.dataset.classes
            self.class_to_idx = self.dataset.class_to_idx
        if hasattr(self.dataset, "transform"):
            # keep global classes consistent
            self.transform = self.dataset.transform
        self.target_transform = target_transform
        self.target_pos = target_pos

    def __getitem__(self, idx):
        items = self.dataset[idx]

        # Ensure it's a tuple
        if not isinstance(items, tuple):
            items = (items,)

        if self.target_transform is not None and len(items) >= self.target_pos:
            items = list(items)
            target = items[-self.target_pos] 
            items[-self.target_pos] = self.target_transform(target)
            items = tuple(items)

        return items

    def __len__(self):
        return len(self.dataset)


def prepare_datasets(root, environments, descriptors, holdout_fraction, seed):
    #environments = [f.name for f in os.scandir(root) if f.is_dir()]
    environments = sorted(environments)
    num_envs = len(environments)
       
    # persistent across descriptors
    in_idxs = [None for _ in range(num_envs)]
    out_idxs = [None for _ in range(num_envs)]
    datasets_list = []
    
    for descriptor in descriptors:
        in_splits = [None for _ in range(num_envs)]
        out_splits = [None for _ in range(num_envs)]
        for env_idx, env_name in enumerate(environments):
            env_dir = os.path.join(root, env_name)
    
            target_transform = descriptor['target_transform'] if not descriptor['wrap'] else None
            env = descriptor['dataset'](env_dir, class_to_idx=descriptor['class_to_index'], 
                    transform=descriptor['transform'], target_transform=target_transform)

            # Split each env (dataset) into an 'in-split' and an 'out-split' dataset. 
            # We'll train on each 'in-split' except the test envs, and evaluate on both splits.
            # We need in_/out_idx to make splits consistent across descriptors
                
            in_split, in_idx, out_split, out_idx = split_dataset(env,
                int(len(env)*holdout_fraction), seed, in_idx=in_idxs[env_idx], out_idx=out_idxs[env_idx])
            in_splits[env_idx] = in_split
            if in_idxs[env_idx] is None:
                in_idxs[env_idx] = in_idx
            out_splits[env_idx] = out_split  
            if out_idxs[env_idx] is None:
                out_idxs[env_idx] = out_idx
        # end for env_idx, env_name in enumerate(environments)
        
        if descriptor['required_split'] == "in":
            splits = in_splits
        else:
            splits = out_splits
        
        dataset = [ConcatDataset(splits)] if len(splits)>1 else splits
        
        if descriptor['wrap']:
            dataset = [TargetTransformWrapper(d, target_transform=target_transform, target_pos=descriptor['target_pos']) for d in dataset]
            
        datasets_list.append(dataset)
    # end for descriptor in descriptors
    
    return datasets_list

def traverse_objects(obj, level=0):
    prefix = '\t'*level
    print(prefix, type(obj).__name__)
    if isinstance(obj, list):
        return [traverse_objects(o, level+1) for o in obj]
    if isinstance(obj, tuple):
        return [traverse_objects(o, level+1) for o in obj]
    elif isinstance(obj, TargetTransformWrapper):
        return traverse_objects(obj.dataset, level+1)
    elif isinstance(obj, ConcatDataset):
        return traverse_objects(obj.datasets, level+1)
    elif isinstance(obj, _SplitDataset):
        return traverse_objects(obj.underlying_dataset, level+1)
    elif isinstance(obj, ImageFolder):
        print(prefix+'\troot:', obj.root)
        return None
    else:
        print(obj)
        return None
       
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Global args
    parser.add_argument('--environments', type=str, nargs='+')
    parser.add_argument('--output_dir', type=str, default="./data/DataSets/CMNIST2/64/")
    parser.add_argument('--holdout_fraction', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--target_image_size', type=int, default=64)
    parser.add_argument('--norandgray', action="store_true")
    parser.add_argument('--target_transform', type=str, default=None, help='a function definition to apply to target')
    parser.add_argument('--class_to_idx', type=str, default=None, help='a function definition to apply to class to obtain it index')
    parser.add_argument('--out', action="store_true")
    parser.add_argument('--wrap', action="store_true")
    

    args = parser.parse_args()

    target_transform = eval(args.target_transform) if args.target_transform is not None else None
    train_transform = utils.make_train_transform(args.target_image_size, randgray=not args.norandgray)
    class_to_idx = eval(args.class_to_idx) if args.class_to_idx is not None else None

    descriptors = [{'dataset': utils.Imagenet_idx_pair,
                    'transform': train_transform,
                    'target_transform': target_transform,
                    'class_to_index': class_to_idx,
                    'wrap': args.wrap, # for changeable target transform
                    'target_pos': 2,
                    'required_split': "in" if not args.out else "out",
    }]
    datasets = prepare_datasets(args.output_dir, args.environments, descriptors, args.holdout_fraction, args.seed)
    traverse_objects(datasets)
            


    