import argparse
import os
## Progress bar
from tqdm.auto import tqdm
import numpy as np
import random
import shutil
from functools import partial
import math
from pathlib import Path

def count_domains(root, domain_names):
    # discover domains and classes first
    # Returns:
    #   domains - list
    #   classes - set
    #   counts - numpy array (domain, class)
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


# terra_inc_counts_simple_numpy.py
#
# Deterministic, non-LP dataset split for TerraInc.
# NumPy-only implementation.
# ASCII only.
#
# PURPOSE
# -------
# Split RAW[d,c] sample counts into:
#   - R_train[d,c] : representation training
#   - R_val[d,c]   : representation validation
#   - P_train[d,c] : LP training (subset of R_train)
#   - P_val[d,c]   : LP validation (equal to R_val)
#
# using a fixed train/val split and per-class LP balancing.
#
# No optimization solver is used.
# No samples are discarded.
# No leakage is possible by construction.
#
# ------------------------------------------------------------
# VARIABLE CONVENTIONS
# ------------------------------------------------------------
#
# d : domain index, size = D
# c : class index,  size = C
#
# RAW[d,c]      : int >= 0, total samples
# R_train[d,c]  : int >= 0
# R_val[d,c]    : int >= 0
# P_train[d,c]  : int >= 0
# P_val[d,c]    : int >= 0
#
# All arrays have shape (D, C)
#
# ------------------------------------------------------------

# ============================================================
# USER CONFIGURATION (ONLY EDIT THIS SECTION)
# ============================================================

def prune_domains(domains, classes, raw, train_fraction=0.8, lp_train_target_per_class=100, do_trim=True):
    DOMAINS = domains
    CLASSES = classes

    D = len(DOMAINS)
    C = len(CLASSES)

    # RAW[d,c] sample counts
    RAW = raw

    TRAIN_FRACTION = train_fraction
    LP_TRAIN_TARGET_PER_CLASS = lp_train_target_per_class

    # ============================================================
    # INPUT VALIDATION
    # ============================================================

    assert RAW.shape == (D, C)
    assert np.all(RAW >= 0)
    assert 0.0 < TRAIN_FRACTION < 1.0

    # ============================================================
    # STEP 1: FIXED TRAIN / VAL SPLIT
    # ============================================================

    R_train = np.zeros((D, C), dtype=int)
    R_val   = np.zeros((D, C), dtype=int)

    for d in range(D):
        for c in range(C):
            train_cnt = int(math.floor(TRAIN_FRACTION * RAW[d, c]))
            val_cnt   = RAW[d, c] - train_cnt
            R_train[d, c] = train_cnt
            R_val[d, c]   = val_cnt

    # LP validation equals representation validation
    P_val = R_val.copy()

    # ============================================================
    # STEP 2: INITIALIZE LP TRAIN
    # ============================================================

    P_train = R_train.copy()

    # ============================================================
    # STEP 3: FAST PER-CLASS LP BALANCING (WITH SAFE TIE HANDLING)
    # ============================================================

    effective_M = np.zeros(C, dtype=int)

    for c in range(C):
        total_available = int(R_train[:, c].sum())
        M_c = min(LP_TRAIN_TARGET_PER_CLASS, total_available)
        effective_M[c] = M_c

    if do_trim:
        for c in range(C):
            while True:
                current_sum = int(P_train[:, c].sum())
                if current_sum <= M_c:
                    break

                # Domains sorted by descending P_train for this class
                order = np.argsort(-P_train[:, c])
                d_max = int(order[0])
                d_2nd = int(order[1])

                max_val = P_train[d_max, c]
                second_val = P_train[d_2nd, c]
                excess = current_sum - M_c

                if max_val == second_val:
                    # Tie case: decrement by exactly 1 to avoid complexity
                    P_train[d_max, c] -= 1
                else:
                    # Fast path: reduce toward second-largest
                    gap = max_val - second_val
                    delta = min(gap, excess)
                    P_train[d_max, c] -= delta

                if P_train[d_max, c] < 0:
                    raise RuntimeError("Negative P_train encountered")

        # ============================================================
        # FINAL CONSISTENCY CHECKS
        # ============================================================

        # Data conservation
        assert np.all(R_train + R_val == RAW)

        # Subset constraints
        assert np.all(P_train <= R_train)
        assert np.all(P_val == R_val)

        # Exact per-class LP totals
        for c in range(C):
            assert int(P_train[:, c].sum()) == effective_M[c]

    # ============================================================
    # HEADROOM ANALYSIS FOR M
    # ============================================================

    # Maximum feasible LP-train per class given the fixed split
    max_M_per_class = R_train.sum(axis=0)   # shape (C,)

    # Per-class headroom relative to user target
    headroom_per_class = max_M_per_class - effective_M

    # Global limiting class
    global_max_M = int(max_M_per_class.min())
    global_headroom = global_max_M - LP_TRAIN_TARGET_PER_CLASS

    # ============================================================
    # OUTPUT
    # ============================================================

    def print_table(name, X):
        print(name + ":")
        header = "      " + " ".join(f"{c:>9}" for c in CLASSES)
        print(header)
        for d in range(D):
            row = f"{DOMAINS[d]:>5} "
            row += " ".join(f"{int(X[d,c]):9d}" for c in range(C))
            print(row)
        print()

    print("TerraInc counts summary")
    print("-----------------------\n")

    print_table("RAW", RAW)
    print_table("P_val", P_val)
    print_table("R_val", R_val)
    print_table("P_train", P_train)
    print_table("R_train", R_train)

    print("Effective LP train target per class:")
    for c in range(C):
        print(f"  {CLASSES[c]}: {effective_M[c]}")

    print("\nAll invariants satisfied.")

    print("\nLP train headroom analysis")
    print("--------------------------")

    for c in range(C):
        status = "USER-LIMITED" if headroom_per_class[c] > 0 else "DATA-LIMITED"
        print(
            f"{CLASSES[c]:>9} | "
            f"max feasible M = {int(max_M_per_class[c]):5d} | "
            f"used M = {effective_M[c]:5d} | "
            f"headroom = {int(headroom_per_class[c]):5d} | "
            f"{status}"
        )

    print("\nGlobal M analysis")
    print("-----------------")
    print("User-selected M:     ", LP_TRAIN_TARGET_PER_CLASS)
    print("Max feasible global M:", global_max_M)
    print("Global headroom:     ", global_headroom)
    
    return P_train, P_val, R_train, R_val

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    input_dir = os.path.join(args.input_dir, str(args.target_image_size) + '/')
    output_dir = os.path.join(args.output_dir, str(args.target_image_size) + '/')
    save_dir_R_train = output_dir + 'R/train/'
    os.makedirs(save_dir_R_train, exist_ok=True)
    save_dir_P_train = output_dir + 'P/train/'
    os.makedirs(save_dir_P_train, exist_ok=True)
    save_dir_R_val = output_dir + 'R/val/'
    os.makedirs(save_dir_R_val, exist_ok=True)
    save_dir_P_val = output_dir + 'P/val/'
    os.makedirs(save_dir_P_val, exist_ok=True)
    save_dir_R_test = output_dir + 'R/test/'
    os.makedirs(save_dir_R_test, exist_ok=True)
    save_dir_P_test = output_dir + 'P/test/'
    os.makedirs(save_dir_P_test, exist_ok=True)
    
    # count number of samples in each class and training domain
    domains, classes, counts = count_domains(input_dir, set(args.domain_names)-set([args.test_domain])) 
    if args.count_only:
        print('Domains:')
        print(domains)
        print('Classes:')
        print(classes)
        print('Counts:')
        print(counts)
        exit(1)
    P_train, P_val, R_train, R_val = prune_domains(domains, classes, raw, train_fraction=0.8, lp_train_target_per_class=100, do_trim=args.balance_counts)

    if args.select_method == 'train':
        with os.scandir(input_dir) as e:
            for env_dir in e:   # env_dir is directory of per-label sub-directories
                if env_dir.name not in args.domain_names:
                    continue
                if env_dir.name != args.test_domain:
                    env_idx = domains.index(env_dir.name)
                    with os.scandir(env_dir) as l:
                        for lab_dir in l:   # lab_dir is a label sub-directory
                            if lab_dir.is_dir():
                                label = lab_dir.name
                                label_idx = classes.index(label)
                                with os.scandir(lab_dir) as fs:     # fs are the images of a label
                                    files = [f for f in fs if f.is_file()]
                                    num_files = len(files)
                                    f_idx = np.random.permutation(num_files)
                                    
                                    # R Train
                                    train_num = R_train[env_idx, label_idx]
                                    train_idx = f_idx[:train_num]
                                    
                                    output_lab_dir = os.path.join(save_dir_R_train, label + '/')
                                    os.makedirs(output_lab_dir, exist_ok=True)                                    
                                    for fp in [files[i] for i in train_idx]:
                                        src = Path(fp.path)
                                        dst = os.path.join(output_lab_dir, fp.name)
                                        dst = Path(dst)                                    
                                        dst.symlink_to(src)
                                    
                                    # R Val
                                    val_num = R_val[env_idx, label_idx]
                                    val_idx = f_idx[train_num:train_num+val_num]
                                    
                                    output_lab_dir = os.path.join(save_dir_R_val, label + '/')
                                    os.makedirs(output_lab_dir, exist_ok=True)
                                    for fp in [files[i] for i in val_idx]:
                                        src = Path(fp.path)
                                        dst = os.path.join(output_lab_dir, fp.name)
                                        dst = Path(dst)                                    
                                        dst.symlink_to(src)

                                    # P Train
                                    train_num = P_train[env_idx, label_idx]
                                    train_idx = f_idx[:train_num]
                                    
                                    output_lab_dir = os.path.join(save_dir_P_train, label + '/')
                                    os.makedirs(output_lab_dir, exist_ok=True)                                    
                                    for fp in [files[i] for i in train_idx]:
                                        src = Path(fp.path)
                                        dst = os.path.join(output_lab_dir, fp.name)
                                        dst = Path(dst)                                    
                                        dst.symlink_to(src)
                                    
                                    # P Val
                                    val_num = P_val[env_idx, label_idx]
                                    val_idx = f_idx[train_num:train_num+val_num]
                                    
                                    output_lab_dir = os.path.join(save_dir_P_val, label + '/')
                                    os.makedirs(output_lab_dir, exist_ok=True)
                                    for fp in [files[i] for i in val_idx]:
                                        src = Path(fp.path)
                                        dst = os.path.join(output_lab_dir, fp.name)
                                        dst = Path(dst)                                    
                                        dst.symlink_to(src)
                else:
                    with os.scandir(env_dir) as l:
                        for lab_dir in l:   # lab_dir is a label sub-directory
                            if lab_dir.is_dir():
                                label = lab_dir.name
                                with os.scandir(lab_dir) as fs:     # fs are the images of a label
                                    files = [f for f in fs if f.is_file()]
                                    output_lab_dir_R = os.path.join(save_dir_R_test, label + '/')
                                    os.makedirs(output_lab_dir_R, exist_ok=True)                                    
                                    output_lab_dir_P = os.path.join(save_dir_P_test, label + '/')
                                    os.makedirs(output_lab_dir_P, exist_ok=True)                                    
                                    for fp in files:
                                        src = Path(fp.path)
                                        # R Test
                                        dst = os.path.join(output_lab_dir_R, fp.name)
                                        dst = Path(dst)                                    
                                        dst.symlink_to(src)
                                        # P Test
                                        dst = os.path.join(output_lab_dir_P, fp.name)
                                        dst = Path(dst)                                    
                                        dst.symlink_to(src)
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
    parser.add_argument('--count_only', action='store_true', help='Only count domains')
    parser.add_argument('--balance_counts', action='store_true', help='Balance counts')

    subparsers = parser.add_subparsers(dest='select_method', required=True)

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--train_split', type=partial(bounded_type, min_val=0.0, max_val=1.0, cast_type=float), required=True)

    loo_parser = subparsers.add_parser('loo')
    loo_parser.add_argument('--val_domain', type=str, required=True)

    args = parser.parse_args()
    
    main(args)


