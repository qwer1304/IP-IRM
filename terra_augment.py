"""
terra_augment.py
================
Generate synthetic images by histogram matching.
Loads pre-built index from terra_analysis.py output.

Usage
-----
python terra_augment.py \
    --index          terra_index.pkl \
    --idir           /path/to/images \
    --odir           /path/to/synthetics \
    --species        bobcat squirrel bird \
    --target_locs    38 43 46 \
    --source_locs    43 46 \
    --N_target       50 \
    --N_synthetic    1 \
    --N_hist_clusters 5 \
    --workers        4 \
    --seed           42 \
    --debug

Design
------
For each (species, target_location):
  - source images   : majority locations, one frame per burst (middle frame)
  - target histogram: background pixels (outside bbox) at target_location,
                      any species, TOD-stratified
  - synthetic       : src animal pixels (inside src_bbox) -> unchanged
                      src background pixels (outside src_bbox) -> recolored to target appearance
  - fallback        : if bbox missing, apply mapping to full image

DG constraint: test location (e.g. L100) is NEVER used. All operations
               are strictly within the provided train locations.

Performance: embarrassingly parallel per source image.
             Use --workers to tune for your environment.
             GPU not used - histogram matching is CPU/IO bound.
"""

import os
import random
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
#from skimage import io
import imageio.v3 as iio

from terra_analysis import (
    load_index,
    print_summary,
    build_summary,
    DEFAULT_LOCATIONS,
    DEFAULT_CATEGORIES,
    CATEGORY_INDEX,
    cat_label,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc  = kwargs.get('desc', '')
        items = list(iterable)
        n     = len(items)
        for i, item in enumerate(items):
            if n > 0 and i % max(1, n // 10) == 0:
                print(f"  {desc}: {i}/{n}")
            yield item

try:
    from sklearn.cluster import KMeans
except ImportError:
    class KMeans:
        def __init__(self, n_clusters, random_state=42, n_init='auto'):
            self.n_clusters   = n_clusters
            self.random_state = random_state
        def fit_predict(self, X):
            rng     = np.random.RandomState(self.random_state)
            idx     = rng.choice(len(X), self.n_clusters, replace=False)
            centers = X[idx].copy()
            labels  = np.zeros(len(X), dtype=int)
            for _ in range(100):
                dists  = np.array([np.sum((X - c) ** 2, axis=1) for c in centers])
                labels = np.argmin(dists, axis=0)
                new_centers = np.array([
                    X[labels == k].mean(axis=0) if (labels == k).any() else centers[k]
                    for k in range(self.n_clusters)
                ])
                if np.allclose(centers, new_centers):
                    break
                centers = new_centers
            return labels


# -- Image loading -------------------------------------------------------------

def load_image(path):
    """Load image as uint8 RGB array."""
    # Replace img = io.imread(path)
    #img = io.imread(path)
    img = iio.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img.astype(np.uint8)


def get_path(iid, image_index, images_root):
    rec = image_index[iid]
    # Directory layout: <images_root>/L<loc>/<species>/<uuid>.jpg
    # annotations is guaranteed non-empty (build_index excludes unannotated images)
    species = rec['annotations'][0]['category']
    return os.path.join(images_root, f"L{rec['location']}", species, rec['file_name'])


# -- Burst handling ------------------------------------------------------------

def get_middle_frame(seq_id, seq_to_images, image_index):
    """Return image_id of the middle frame in a burst."""
    frames = seq_to_images[seq_id]   # already sorted by frame_num
    return frames[len(frames) // 2]


# -- Bbox utilities ------------------------------------------------------------

def bbox_to_mask(bbox, h, w, ann_h=None, ann_w=None):
    """
    Convert [x, y, bw, bh] bbox to boolean mask of shape (h, w).
    True = inside bbox (animal region). Returns None if bbox is None.

    ann_h, ann_w: annotation-file image dimensions (may differ from actual
                  loaded image if files were downsampled). When provided, bbox
                  coordinates are scaled proportionally to match actual (h, w).
    """
    if bbox is None:
        return None
    x_orig, y_orig, bw_orig, bh_orig = bbox
    if ann_h is not None and ann_w is not None and ann_h > 0 and ann_w > 0:
        scaled = True
        scale_x = w / ann_w
        scale_y = h / ann_h
        x, bw = x_orig * scale_x, bw_orig * scale_x
        y, bh = y_orig * scale_y, bh_orig * scale_y
    else:
        scaled = False
        x, y, bw, bh = bbox
    
    x, y, bw, bh = int(round(x)), int(round(y)), int(round(bw)), int(round(bh))
    # negative origin is always a genuine error
    assert x >= 0 and y >= 0, (
        "bbox origin (%d, %d) is negative -- corrupt bbox or bad annotation dims, scaled %d" % (x, y, scaled)
    )
    # zero/negative size is always a genuine error
    assert bw > 0 and bh > 0, (
        "bbox has zero or negative size (%dx%d) scaled %d" % (bw, bh, scaled)
    )
    # allow up to 2px rounding slop from scaling; anything larger is a
    # genuine annotation/file mismatch and should stop execution
    x_over = (x + bw) - w
    y_over = (y + bh) - h
    assert x_over <= 2 and y_over <= 2, (
        "bbox orig (%d,%d,%d,%d) bbox scaled (%d,%d,%d,%d) exceeds image (%dx%d) by (%d,%d) px -- "
        "annotation dims (%dx%d) do not match actual file"
        % (x_orig, y_orig, bw_orig, bh_orig, x, y, bw, bh, w, h, x_over, y_over, ann_w, ann_h)
    )
    x1, y1 = max(0, x),      max(0, y)
    x2, y2 = min(w, x + bw), min(h, y + bh)
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


# -- Histogram utilities -------------------------------------------------------

def compute_bg_histogram(img, bbox, n_bins=256, ann_h=None, ann_w=None):
    """
    Compute per-channel histogram using BACKGROUND pixels only (outside bbox).
    Falls back to full image if bbox is None or covers entire image.
    Returns array of shape (3, n_bins).

    ann_h, ann_w: annotation-file image dimensions for bbox scaling.
    """
    h, w        = img.shape[:2]
    animal_mask = bbox_to_mask(bbox, h, w, ann_h=ann_h, ann_w=ann_w)

    if animal_mask is not None and animal_mask.any() and not animal_mask.all():
        bg_mask = ~animal_mask
    else:
        bg_mask = np.ones((h, w), dtype=bool)

    hists = []
    for ch in range(3):
        pixels = img[:, :, ch][bg_mask]
        hist, _ = np.histogram(pixels, bins=n_bins, range=(0, 256))
        hists.append(hist)
    return np.array(hists, dtype=np.float32)


def average_bg_histograms(image_ids, image_index, images_root, n_bins=256,
                          verbose=False):
    """Average background histograms across a list of image_ids."""
    hists      = []
    n_failed   = 0
    first_fail = None
    for iid in image_ids:
        rec  = image_index[iid]
        path = get_path(iid, image_index, images_root)
        try:
            img = load_image(path)
        except Exception as e:
            n_failed += 1
            if first_fail is None:
                first_fail = (path, str(e))
            if verbose: print(f"  WARNING: could not load {path}: {e}")
            continue
        bbox  = rec['annotations'][0]['bbox'] if rec['annotations'] else None
        ann_h = rec.get('height')
        ann_w = rec.get('width')
        hists.append(compute_bg_histogram(img, bbox, n_bins, ann_h=ann_h, ann_w=ann_w))
    if not hists:
        if first_fail is not None:
            raise FileNotFoundError(
                f"Failed to load ANY of {n_failed} images. "
                f"First failure: {first_fail[0]}: {first_fail[1]}"
            )
        return None
    if n_failed > 0 and verbose:
        print(f"  WARNING: {n_failed}/{n_failed + len(hists)} images failed to load")
    return np.mean(hists, axis=0)   # (3, n_bins)


def cluster_bg_histograms(image_ids, image_index, images_root,
                          n_clusters, n_bins=256, seed=42, verbose=False):
    """
    Cluster images by background histogram similarity.
    Returns list of n_clusters representative average histograms.
    """
    hist_list  = []
    valid_ids  = []
    n_failed   = 0
    first_fail = None
    for iid in image_ids:
        rec  = image_index[iid]
        path = get_path(iid, image_index, images_root)
        try:
            img = load_image(path)
        except Exception as e:
            n_failed += 1
            if first_fail is None:
                first_fail = (path, str(e))
            continue
        bbox  = rec['annotations'][0]['bbox'] if rec['annotations'] else None
        ann_h = rec.get('height')
        ann_w = rec.get('width')
        hist_list.append(compute_bg_histogram(img, bbox, n_bins, ann_h=ann_h, ann_w=ann_w).flatten())
        valid_ids.append(iid)

    if not hist_list:
        if first_fail is not None:
            raise FileNotFoundError(
                f"Failed to load ANY of {n_failed} images for clustering. "
                f"First failure: {first_fail[0]}: {first_fail[1]}"
            )
        return []

    n_clusters = min(n_clusters, len(hist_list))
    X      = np.array(hist_list)
    km     = KMeans(n_clusters=n_clusters, random_state=seed)
    labels = km.fit_predict(X)

    rep_hists = []
    for c in range(n_clusters):
        cluster_ids = [valid_ids[i] for i, l in enumerate(labels) if l == c]
        h = average_bg_histograms(cluster_ids, image_index, images_root,
                                  n_bins, verbose=verbose)
        if h is not None:
            rep_hists.append(h)
    return rep_hists   # list of (3, n_bins)


# -- CDF mapping ---------------------------------------------------------------

def compute_cdf_mapping(src_pixels_per_channel, target_hist):
    """
    Compute per-channel pixel value mapping via CDF matching.

    src_pixels_per_channel : list of 3 x 1D arrays of pixel values
    target_hist            : (3, n_bins) array

    Returns list of 3 lookup tables (uint8 arrays of length 256).
    """
    mappings = []
    for ch in range(3):
        src_hist, _ = np.histogram(src_pixels_per_channel[ch],
                                   bins=256, range=(0, 256))
        src_cdf  = np.cumsum(src_hist).astype(float)
        src_cdf /= src_cdf[-1] + 1e-8

        tgt_cdf  = np.cumsum(target_hist[ch]).astype(float)
        tgt_cdf /= tgt_cdf[-1] + 1e-8

        mapping = np.zeros(256, dtype=np.uint8)
        j = 0
        for i in range(256):
            while j < 255 and tgt_cdf[j] < src_cdf[i]:
                j += 1
            mapping[i] = j
        mappings.append(mapping)
    return mappings


def apply_mapping(src_img, mappings, animal_mask=None):
    """
    Apply per-channel pixel mapping to src_img.
    If animal_mask provided:
      - BACKGROUND (outside mask): remapped to target appearance
      - ANIMAL (inside mask): left unchanged (src animal preserved)
    If no animal_mask: apply to full image (fallback).
    """
    result = src_img.copy()
    for ch in range(3):
        channel = src_img[:, :, ch]
        mapped  = mappings[ch][channel]
        if animal_mask is not None:
            result[:, :, ch][~animal_mask] = mapped[~animal_mask]
        else:
            result[:, :, ch] = mapped
    return result


# -- Target histogram builder --------------------------------------------------

def build_target_histograms(target_location, image_index, loc_cat_to_seqs,
                             seq_to_images, images_root,
                             N_TARGET, N_SYNTHETIC, N_HIST_CLUSTERS,
                             STRATIFY_TOD, seed, verbose=False):
    """
    Build target background histogram(s) for target_location.
    Uses ALL species at target_location (any species gives location appearance).
    One frame per burst to avoid near-duplicates.
    Returns dict: stratum -> list of (3, n_bins) histograms.
    """
    all_ids = []
    for cat in DEFAULT_CATEGORIES:
        seqs = loc_cat_to_seqs.get((target_location, cat), [])
        for seq_id in seqs:
            all_ids.append(get_middle_frame(seq_id, seq_to_images, image_index))
    all_ids = list(set(all_ids))

    random.seed(seed)
    strata = ['day', 'night'] if STRATIFY_TOD else ['all']
    target_hists = {}

    for stratum in strata:
        pool = all_ids if stratum == 'all' else \
               [iid for iid in all_ids if image_index[iid]['tod'] == stratum]

        if not pool:
            if verbose:
                print(f"  WARNING: no {stratum} images at L{target_location} for histogram")
            target_hists[stratum] = None
            continue

        sampled = random.sample(pool, min(N_TARGET, len(pool)))

        if N_SYNTHETIC == 1:
            h = average_bg_histograms(sampled, image_index, images_root,
                                      verbose=verbose)
            target_hists[stratum] = [h] if h is not None else None
        else:
            hists = cluster_bg_histograms(
                sampled, image_index, images_root,
                n_clusters=min(N_HIST_CLUSTERS, len(sampled)),
                seed=seed, verbose=verbose
            )
            target_hists[stratum] = hists if hists else None

    return target_hists


# -- Per-image worker (top-level for multiprocessing pickling) -----------------

def _process_one(args):
    """
    Worker function: process a single source image.
    args is a dict to avoid multiprocessing pickling issues with many positional args.
    Returns number of images generated (0 or N_SYNTHETIC).
    """
    iid            = args['iid']
    image_index    = args['image_index']
    images_root    = args['images_root']
    out_folder     = args['out_folder']
    target_hists   = args['target_hists']
    target_location= args['target_location']
    species        = args['species']
    N_SYNTHETIC    = args['N_SYNTHETIC']
    STRATIFY_TOD   = args['STRATIFY_TOD']
    verbose        = args['verbose']

    rec     = image_index[iid]
    stratum = rec['tod'] if STRATIFY_TOD else 'all'

    hists = target_hists.get(stratum)
    if not hists:
        fallback = 'night' if stratum == 'day' else 'day'
        hists = target_hists.get(fallback)
    if not hists:
        return 0

    selected_hists = random.sample(hists, min(N_SYNTHETIC, len(hists)))

    src_path = get_path(iid, image_index, images_root)
    try:
        src_img = load_image(src_path)
    except Exception as e:
        if verbose: print(f"  ERROR loading {src_path}: {e}")
        return 0

    h, w = src_img.shape[:2]
    src_bbox = next(
        (ann['bbox'] for ann in rec['annotations'] if ann['category'] == species),
        None
    )
    ann_h = rec.get('height')
    ann_w = rec.get('width')
    animal_mask = bbox_to_mask(src_bbox, h, w, ann_h=ann_h, ann_w=ann_w)

    if animal_mask is not None and animal_mask.any() and not animal_mask.all():
        bg_mask = ~animal_mask
    else:
        bg_mask = np.ones((h, w), dtype=bool)

    src_bg_pixels = [src_img[:, :, ch][bg_mask] for ch in range(3)]

    n_generated = 0
    for idx, target_hist in enumerate(selected_hists):
        try:
            mappings  = compute_cdf_mapping(src_bg_pixels, target_hist)
            synthetic = apply_mapping(src_img, mappings, animal_mask)
            out_fname = f"syn_{Path(rec['file_name']).stem}_s{idx}.jpg"
            out_path  = os.path.join(out_folder, out_fname)
            # Replace io.imsave(out_path, synthetic, quality=90)
            #io.imsave(out_path, synthetic, quality=90)
            iio.imwrite(out_path, synthetic, quality=90)
            n_generated += 1
        except Exception as e:
            if verbose: print(f"  ERROR generating synthetic for {src_path}: {e}")

    return n_generated


# -- Core augmentation ---------------------------------------------------------

def augment_minority_species(
    species,
    target_location,
    source_locations,
    image_index,
    loc_cat_to_images,
    loc_cat_to_seqs,
    seq_to_images,
    images_root,
    output_root,
    train_locations,
    N_TARGET         = 50,
    N_SYNTHETIC      = 1,
    N_HIST_CLUSTERS  = 5,
    STRATIFY_TOD     = True,
    workers                = 4,
    seed                   = 42,
    verbose                = False,
    dry_run                = False,
    use_whole_src_seq_thresh = 20,
):
    """
    Generate synthetic images of `species` at `target_location`.

    Parameters
    ----------
    species                  : str        e.g. 'bobcat'
    target_location          : int        minority train location e.g. 38
    source_locations         : list[int]  majority train locations e.g. [43, 46]
    images_root              : str        root folder of original images
    output_root              : str        root folder for synthetic output
    train_locations          : set[int]   allowed train locations (DG guard)
    N_TARGET                 : int        target images for histogram building
    N_SYNTHETIC              : int        synthetics per source image
    N_HIST_CLUSTERS          : int        histogram clusters when N_SYNTHETIC > 1
    STRATIFY_TOD             : bool       separate day / night histograms
    workers                  : int        number of parallel worker processes
    seed                     : int        random seed
    verbose                  : bool       print progress messages
    dry_run                  : bool       count expected outputs without writing anything
    use_whole_src_seq_thresh : int        if total source sequences < this threshold,
                                          use ALL burst frames instead of middle frame only
    """
    assert target_location in train_locations, \
        f"target_location {target_location} not in train_locations {train_locations}"
    assert all(l in train_locations for l in source_locations), \
        f"source_locations must be within train_locations {train_locations}"

    # -- collect source images ------------------------------------------------
    all_seqs = []
    for src_loc in source_locations:
        all_seqs.extend(loc_cat_to_seqs.get((src_loc, species), []))

    use_all_frames = len(all_seqs) < use_whole_src_seq_thresh

    source_ids = []
    for seq_id in all_seqs:
        if use_all_frames:
            source_ids.extend(seq_to_images[seq_id])   # all frames
        else:
            source_ids.append(get_middle_frame(seq_id, seq_to_images, image_index))

    if verbose:
        mode = "all frames" if use_all_frames else "middle frame"
        print(f"  {len(all_seqs)} source seqs ({mode}), {len(source_ids)} source images")

    if not source_ids:
        if verbose: print(f"  No source images for {species} at {source_locations}")
        return 0

    n_expected = len(source_ids) * N_SYNTHETIC

    if dry_run:
        print(f"  [DRY RUN] {species}@L{target_location}: "
              f"{len(source_ids)} source seqs x {N_SYNTHETIC} = {n_expected} synthetics")
        return n_expected

    random.seed(seed)
    np.random.seed(seed)

    # -- Step 1: build target background histograms ---------------------------
    if verbose:
        print(f"\n[{species}@L{target_location}] Building target histograms ...")
    target_hists = build_target_histograms(
        target_location, image_index, loc_cat_to_seqs,
        seq_to_images, images_root,
        N_TARGET, N_SYNTHETIC, N_HIST_CLUSTERS, STRATIFY_TOD, seed,
        verbose=verbose
    )

    if verbose: print(f"  {len(source_ids)} source images (one per burst)")

    # -- Step 2: output folder ------------------------------------------------
    out_folder = os.path.join(output_root, f'L{target_location}', species)
    os.makedirs(out_folder, exist_ok=True)

    # -- Step 3: build per-image arg dicts for workers ------------------------
    work_items = [
        {
            'iid':             iid,
            'image_index':     image_index,
            'images_root':     images_root,
            'out_folder':      out_folder,
            'target_hists':    target_hists,
            'target_location': target_location,
            'species':         species,
            'N_SYNTHETIC':     N_SYNTHETIC,
            'STRATIFY_TOD':    STRATIFY_TOD,
            'verbose':         verbose,
        }
        for i, iid in enumerate(source_ids)
    ]

    # -- Step 4: process in parallel (threads: I/O-bound, no pickle overhead) --
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(tqdm(
                ex.map(_process_one, work_items),
                total=len(work_items),
                desc=f"{species}@L{target_location}"
            ))
    else:
        results = [
            _process_one(item)
            for item in tqdm(work_items, desc=f"{species}@L{target_location}")
        ]

    n_generated = sum(results)
    if verbose:
        print(f"  Generated {n_generated} synthetic images -> {out_folder}")
    return n_generated


# -- Dry run utilities ---------------------------------------------------------

def dry_run_counts(species_list, target_locs, source_locs, loc_cat_to_seqs,
                   seq_to_images, image_index, N_SYNTHETIC,
                   use_whole_src_seq_thresh=20):
    """
    Compute expected synthetic counts per (location, species) without
    touching disk or building histograms.
    Returns dict: (location, species) -> n_synthetics
    """
    counts   = {}
    src_seqs = {}   # (tgt_loc, species) -> n source sequences
    for tgt_loc in target_locs:
        src_locs = [l for l in source_locs if l != tgt_loc]
        for species in species_list:
            all_seqs = []
            for src in src_locs:
                all_seqs.extend(loc_cat_to_seqs.get((src, species), []))
            if not all_seqs:
                continue
            src_seqs[(tgt_loc, species)] = len(all_seqs)
            use_all_frames = len(all_seqs) < use_whole_src_seq_thresh
            if use_all_frames:
                n_imgs = sum(len(seq_to_images[s]) for s in all_seqs)
            else:
                n_imgs = len(all_seqs)
            counts[(tgt_loc, species)] = n_imgs * N_SYNTHETIC
    return counts, src_seqs


def print_augmented_summary(summary, syn_counts, src_seqs, include_locations,
                             include_categories):
    """
    Print the three summary tables with synthetic counts overlaid.
    syn_counts: dict (location, species) -> n_synthetics
    """
    locs = sorted(include_locations)
    cats = sorted(include_categories, key=lambda c: CATEGORY_INDEX.get(c, 99))

    # -- table 1: full detail per (loc, cat) ----------------------------------
    print("=" * 115)
    print("PER LOCATION x CATEGORY  (original + synthetic)")
    print(f"{'LOC':<6} {'CATEGORY':<16} {'ORIG':>7} {'SYN':>7} {'TOTAL':>7} "
          f"{'SEQS':>6} {'SEQS_SRC':>9} {'DAY':>6} {'NIGHT':>6} {'BBOX%':>7}")
    print("=" * 115)
    for loc in locs:
        for cat in cats:
            s    = summary.get(f"L{loc}/{cat}")
            syn  = syn_counts.get((loc, cat), 0)
            if s is None and syn == 0:
                continue
            orig     = s['n_images'] if s else 0
            seqs     = s['n_seqs']   if s else 0
            seqs_src = src_seqs.get((loc, cat), 0)
            day      = s['n_day']    if s else 0
            night    = s['n_night']  if s else 0
            bbox     = s['bbox_pct'] if s else 0
            total    = orig + syn
            syn_str  = f"+{syn}" if syn > 0 else "-"
            print(f"L{loc:<5} {cat_label(cat):<16} {orig:>7} {syn_str:>7} {total:>7} "
                  f"{seqs:>6} {seqs_src:>9} {day:>6} {night:>6} {bbox:>6}%")
    print()

    # -- table 2: per species totals -------------------------------------------
    print("=" * 70)
    print("PER SPECIES TOTALS  (original + synthetic)")
    print(f"{'CATEGORY':<16} {'ORIG':>7} {'SYN':>7} {'TOTAL':>7} "
          f"{'SEQS':>6} {'LOCS':>6} {'DAY':>6} {'NIGHT':>6}")
    print("=" * 70)
    for cat in cats:
        orig = n_day = n_night = n_seqs = n_locs = 0
        syn  = 0
        for loc in locs:
            s = summary.get(f"L{loc}/{cat}")
            if s:
                orig    += s['n_images']
                n_day   += s['n_day']
                n_night += s['n_night']
                n_seqs  += s['n_seqs']
                n_locs  += 1
            syn += syn_counts.get((loc, cat), 0)
        if orig + syn == 0:
            continue
        syn_str = f"+{syn}" if syn > 0 else "-"
        print(f"{cat_label(cat):<16} {orig:>7} {syn_str:>7} {orig+syn:>7} "
              f"{n_seqs:>6} {n_locs:>6} {n_day:>6} {n_night:>6}")
    print()

    # -- table 3: species x location matrix -----------------------------------
    col_w = 20
    header = (f"{'CATEGORY':<16}"
              + ''.join(f"{'L'+str(l):>{col_w}}" for l in locs)
              + f"{'TOTAL':>{col_w}}")
    print("=" * len(header))
    print("IMAGES PER SPECIES x LOCATION  (day/night+synthetic)")
    print(header)
    print("=" * len(header))

    for cat in cats:
        cells = []
        for loc in locs:
            s   = summary.get(f"L{loc}/{cat}")
            syn = syn_counts.get((loc, cat), 0)
            d   = s['n_day']   if s else 0
            n   = s['n_night'] if s else 0
            cells.append((d, n, syn))

        if sum(d + n + syn for d, n, syn in cells) == 0:
            continue

        row = f"{cat_label(cat):<16}"
        for d, n, syn in cells:
            if d + n + syn == 0:
                cell = "-"
            elif syn > 0:
                cell = f"{d}/{n}+{syn}"
            else:
                cell = f"{d}/{n}"
            row += f"{cell:>{col_w}}"

        td  = sum(d   for d, n, syn in cells)
        tn  = sum(n   for d, n, syn in cells)
        ts  = sum(syn for d, n, syn in cells)
        tot = f"{td}/{tn}+{ts}" if ts > 0 else f"{td}/{tn}"
        row += f"{tot:>{col_w}}"
        print(row)

    # totals row
    print("-" * len(header))
    tot_row = f"{'TOTAL':<16}"
    gd = gn = gs = 0
    for loc in locs:
        cd = cn = cs = 0
        for cat in cats:
            s   = summary.get(f"L{loc}/{cat}")
            cd += s['n_day']   if s else 0
            cn += s['n_night'] if s else 0
            cs += syn_counts.get((loc, cat), 0)
        cell = f"{cd}/{cn}+{cs}" if cs > 0 else (f"{cd}/{cn}" if cd+cn > 0 else "-")
        tot_row += f"{cell:>{col_w}}"
        gd += cd; gn += cn; gs += cs
    tot = f"{gd}/{gn}+{gs}" if gs > 0 else f"{gd}/{gn}"
    tot_row += f"{tot:>{col_w}}"
    print(tot_row)
    print()


# -- Main ---------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TerraInc histogram matching augmentation')
    parser.add_argument('--index',           default='terra_index.pkl',
                        help='Path to terra_index.pkl (default: terra_index.pkl)')
    parser.add_argument('--idir',            default=None,
                        help='Root folder of original images (required unless --dry_run)')
    parser.add_argument('--odir',            default=None,
                        help='Root folder for synthetic images (required unless --dry_run)')
    parser.add_argument('--species',         nargs='+',
                        default=['bobcat', 'squirrel', 'bird'],
                        help='Species to augment')
    parser.add_argument('--target_locs',     nargs='+', type=int,
                        default=[38],
                        help='Minority train locations to augment toward')
    parser.add_argument('--source_locs',     nargs='+', type=int,
                        default=[43, 46],
                        help='Majority train locations to draw source images from')
    parser.add_argument('--train_locs',      nargs='+', type=int,
                        default=[38, 43, 46],
                        help='All allowed train locations (DG guard)')
    parser.add_argument('--N_target',        type=int, default=50)
    parser.add_argument('--N_synthetic',     type=int, default=1)
    parser.add_argument('--N_hist_clusters', type=int, default=5)
    parser.add_argument('--no_stratify_tod', action='store_true')
    parser.add_argument('--workers',         type=int, default=4,
                        help='Number of parallel worker processes')
    parser.add_argument('--seed',            type=int, default=42)
    parser.add_argument('--use_whole_src_seq_thresh', type=int, default=20,
                        help='Use all burst frames (not just middle) when source '
                             'sequences < this threshold (default: 20)')
    parser.add_argument('--dry_run',         action='store_true',
                        help='Show augmented summary tables without generating images')
    parser.add_argument('--debug',           action='store_true',
                        help='Print progress and warning messages')
    parser.add_argument('--quiet',           action='store_true',
                        help="Don't print summary")
    args = parser.parse_args()

    if not args.dry_run:
        if args.idir is None:
            parser.error('--idir is required unless --dry_run is set')
        if args.odir is None:
            parser.error('--odir is required unless --dry_run is set')
        os.makedirs(args.odir, exist_ok=True)

    train_locations  = set(args.train_locs)
    inc_locs         = set(args.train_locs)

    # load pre-built index
    image_index, loc_cat_to_images, loc_cat_to_seqs, seq_to_images = \
        load_index(args.index, verbose=args.debug)

    summary = build_summary(image_index, loc_cat_to_images,
                            loc_cat_to_seqs, seq_to_images,
                            include_locations=inc_locs)

    if args.dry_run:
        syn_counts = dry_run_counts(  # returns (counts, src_seqs) tuple -- unpacked below
            species_list             = args.species,
            target_locs              = args.target_locs,
            source_locs              = args.source_locs,
            loc_cat_to_seqs          = loc_cat_to_seqs,
            seq_to_images            = seq_to_images,
            image_index              = image_index,
            N_SYNTHETIC              = args.N_synthetic,
            use_whole_src_seq_thresh = args.use_whole_src_seq_thresh,
        )
        syn_counts, src_seqs = syn_counts
        print_augmented_summary(summary, syn_counts, src_seqs, set(args.target_locs), DEFAULT_CATEGORIES)
    else:
        if not args.quiet:
            print_summary(summary)
        total = 0
        for tgt_loc in args.target_locs:
            src_locs = [l for l in args.source_locs if l != tgt_loc]
            for species in args.species:
                n = augment_minority_species(
                    species          = species,
                    target_location  = tgt_loc,
                    source_locations = src_locs,
                    image_index      = image_index,
                    loc_cat_to_images= loc_cat_to_images,
                    loc_cat_to_seqs  = loc_cat_to_seqs,
                    seq_to_images    = seq_to_images,
                    images_root      = args.idir,
                    output_root      = args.odir,
                    train_locations  = train_locations,
                    N_TARGET         = args.N_target,
                    N_SYNTHETIC      = args.N_synthetic,
                    N_HIST_CLUSTERS  = args.N_hist_clusters,
                    STRATIFY_TOD     = not args.no_stratify_tod,
                    workers          = args.workers,
                    seed             = args.seed,
                    verbose          = args.debug,
                    dry_run          = False,
                    use_whole_src_seq_thresh = args.use_whole_src_seq_thresh,
                )
                total += n
        print(f"\nTotal synthetic images generated: {total}")

