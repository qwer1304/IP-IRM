"""
terra_combine.py
================
Create a combined image directory by symlinking original and synthetic images.
Preserves the root/location/class structure expected by downstream tools.

Usage
-----
python terra_combine.py \
    --orig  /path/to/original/images \
    --syn   /path/to/synthetic/images \
    --out   /path/to/combined \
    --debug

The combined dir will contain symlinks to all files from --orig and --syn.
Conflicts (same filename in both): synthetic wins (orig is overridden).
Both --orig and --syn must follow root/L{loc}/{species}/ structure.
--syn may be a subset of locations/species (only augmented ones present).
"""

import os
import argparse
from pathlib import Path


def combine(orig_root, syn_root, out_root, verbose=False):
    """
    Create combined directory with symlinks to orig and syn images.

    Parameters
    ----------
    orig_root : str   root of original images (root/L{loc}/{species}/)
    syn_root  : str   root of synthetic images (same structure, subset)
    out_root  : str   root of combined output (created if needed)
    verbose   : bool  print progress
    """
    orig_root = Path(orig_root).resolve()
    syn_root  = Path(syn_root).resolve()
    out_root  = Path(out_root).resolve()

    if out_root == orig_root or out_root == syn_root:
        raise ValueError("--out must differ from --orig and --syn")

    n_orig = n_syn = n_skip = 0

    # -- pass 1: symlink all original files -----------------------------------
    for src_path in sorted(orig_root.rglob('*')):
        if not src_path.is_file():
            continue
        real_path = src_path.resolve()   # follow any existing symlinks
        rel       = src_path.relative_to(orig_root)
        dst_path  = out_root / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists() or dst_path.is_symlink():
            dst_path.unlink()
        dst_path.symlink_to(real_path)
        n_orig += 1
        if verbose: print(f"  orig -> {rel}")

    # -- pass 2: symlink synthetic files (overrides orig on conflict) ---------
    for src_path in sorted(syn_root.rglob('*')):
        if not src_path.is_file():
            continue
        real_path = src_path.resolve()   # follow any existing symlinks
        rel       = src_path.relative_to(syn_root)
        dst_path  = out_root / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists() or dst_path.is_symlink():
            if verbose: print(f"  OVERRIDE {rel}")
            dst_path.unlink()
        dst_path.symlink_to(real_path)
        n_syn += 1
        if verbose: print(f"  syn  -> {rel}")

    print(f"Combined dir: {out_root}")
    print(f"  Symlinks from orig : {n_orig}")
    print(f"  Symlinks from syn  : {n_syn}")
    print(f"  Total              : {n_orig + n_syn}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combine original and synthetic image dirs via symlinks')
    parser.add_argument('--orig',  required=True,
                        help='Root dir of original images')
    parser.add_argument('--syn',   required=True,
                        help='Root dir of synthetic images')
    parser.add_argument('--out',   required=True,
                        help='Root dir for combined output (symlinks)')
    parser.add_argument('--debug', action='store_true',
                        help='Print each symlink created')
    args = parser.parse_args()

    combine(args.orig, args.syn, args.out, verbose=args.debug)
