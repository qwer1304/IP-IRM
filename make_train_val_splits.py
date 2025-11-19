import argparse
import os
## Progress bar
from tqdm.auto import tqdm
import numpy as np
import random
import shutil
from functools import partial

def count_domains(root, domain_names):
    # discover domains and classes first
    # domains - list
    # classes - set
    # counts - numpy array (domain, class)
    domains = []
    classes = set()

    with os.scandir(root) as domains_iter:
        for d_entry in domains_iter:
            if d_entry.is_dir():
                if d_entry.name not in domain_names:
                    continue
                domains.append(d_entry.name)
                with os.scandir(d_entry.path) as labels_iter:
                    for l_entry in labels_iter:
                        if l_entry.is_dir():
                            classes.add(l_entry.name)

    domains = sorted(domains)
    classes = sorted(classes)

    # mapping: domain -> row, class -> col
    d2i = {d: i for i, d in enumerate(domains)}
    c2i = {c: j for j, c in enumerate(classes)}

    # counts matrix
    counts = np.zeros((len(domains), len(classes)), dtype=int)

    # fill matrix
    with os.scandir(root) as domains_iter:
        for d_entry in domains_iter:
            if not d_entry.is_dir():
                continue
            if d_entry.name not in domain_names:
                continue
            di = d2i[d_entry.name]
            with os.scandir(d_entry.path) as labels_iter:
                for l_entry in labels_iter:
                    if not l_entry.is_dir():
                        continue
                    cj = c2i[l_entry.name]
                    n_images = sum(1 for f in os.scandir(l_entry.path) if f.is_file())
                    counts[di, cj] = n_images
                    
    return domains, classes, counts

def prune_datasets(counts, min_count=0, p=40, extreme_ratio=5, val_fraction=0.2):
    """
    Parameters:
    counts: numpy array, (domain, class)
    min_count: int, remove tiny cells
    p: int, percentile for normal classes
    extreme_ratio, int, threshold to detect extreme dominance - if number of samples in a class in a domain > extreme_ratio*2nd_max_domain, prune to that #
    val_fraction, float, fraction of remaining training samples to use for validation
    Returns:
        balanced_counts - numpy array (domain, class)
        discarded_counts - numpy array (domain, class)
    """

    # ---------------------------
    # Step 1: Extract training domains (exclude L100)
    # ---------------------------
    train_counts = counts.copy()

    # Step 1a: Remove tiny cells
    train_counts_filtered = train_counts.copy()
    train_counts_filtered[train_counts_filtered < min_count] = 0

    # ---------------------------
    # Step 2: Extreme-class handling + percentile capping
    # ---------------------------
    D, C = train_counts_filtered.shape
    balanced_counts = train_counts_filtered.copy()

    for c in range(C):
        cls_counts = train_counts_filtered[:, c]
        if np.all(cls_counts == 0):
            continue  # skip empty class

        max_count = cls_counts.max()
        other_counts = cls_counts[cls_counts != max_count]

        # Extreme-class handling
        if len(other_counts) > 0 and max_count / (other_counts.max()) >= extreme_ratio:
            # Subsample dominant domain to next largest domain count
            second_largest = other_counts.max()
            for d in range(D):
                if cls_counts[d] == max_count:
                    balanced_counts[d, c] = second_largest
        else:
            # Normal class: cap by p-th percentile
            nonzero_counts = cls_counts[cls_counts > 0]
            cap = int(np.percentile(nonzero_counts, p))
            for d in range(D):
                balanced_counts[d, c] = min(cls_counts[d], cap)

    # ---------------------------
    # Step 3: Compute discarded samples (for validation)
    # ---------------------------
    discarded_counts = train_counts - balanced_counts
    discarded_counts[discarded_counts < 0] = 0  # safety

    return balanced_counts, discarded_counts

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
    
    # count number of samples in each class and domain
    domains, classes, counts = count_domains(input_dir, set(args.domain_names)-set([args.test_domain])) 
    # remove test domain
    balanced_counts, discarded_counts = prune_datasets(counts, min_count=args.min_size, val_fraction=1-args.train_split)
    print(balanced_counts)
    print(discarded_counts)

    if args.select_method == 'train':
        with os.scandir(input_dir) as e:      # env_dir is directory of per-label sub-directories
            for env_dir in e:
                if env_dir.name not in args.domain_names:
                    continue
                if env_dir.name != args.test_domain:
                    env_idx = domains.index(env_dir.name)
                    with os.scandir(env_dir) as l:    # lab_dir is a label sub-directory
                        for lab_dir in l:
                            if lab_dir.is_dir():
                                label = lab_dir.name
                                label_idx = classes.index(label)
                                with os.scandir(lab_dir) as fs:     # fs are the images of a label
                                    files = [f for f in fs if f.is_file()]
                                    num_files = len(files)
                                    f_idx = np.random.permutation(num_files)
                                    train_num = balanced_counts[env_idx, label_idx]
                                    train_idx = f_idx[:train_num]
                                    val_idx = f_idx[train_num:]
                                    print(train_num, len(train_idx), len(val_idx))
                                    output_lab_dir = os.path.join(save_dir_train, label + '/')
                                    os.makedirs(output_lab_dir, exist_ok=True)
                                    for fp in [files[i] for i in train_idx]:
                                        shutil.copy(fp, output_lab_dir)
                                    output_lab_dir = os.path.join(save_dir_val, label + '/')
                                    os.makedirs(output_lab_dir, exist_ok=True)
                                    for fp in [files[i] for i in val_idx]:
                                        shutil.copy(fp, output_lab_dir)
                else:
                    shutil.copytree(env_dir, save_dir_test, dirs_exist_ok=True)
    elif args.select_method == 'loo':
        with os.scandir(input_dir) as e:      # env_dir is directory of per-label sub-directories
            for env_dir in e:
                if env_dir.name not in args.domain_names:
                    continue
                if env_dir.is_dir():
                    if env_dir == args.val_dir:
                        output_task_dir = save_dir_val
                    elif env_dir == args.test_dir:
                        output_task_dir = save_dir_test
                    else:
                        output_task_dir = save_dir_train
                    shutil.copytree(env_dir, output_task_dir, dirs_exist_ok=True)

def bounded_type(x, min_val, max_val, cast_type=float):
    try:
        val = cast_type(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} is not a valid {cast_type.__name__}")
    if not (min_val <= val <= max_val):
        raise argparse.ArgumentTypeError(
            f"{val} not in range [{min_val}, {max_val}]"
        )
    return val                                              
            
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
    train_parser.add_argument('--min_size', type=int, default=0, help='min size of training split')

    loo_parser = subparsers.add_parser('loo')
    loo_parser.add_argument('--val_domain', type=str, required=True)

    args = parser.parse_args()
    
    main(args)


