"""
terra_analysis.py
=================
Run ONCE to build and serialize the TerraIncognita annotation index.

Usage
-----
# Use all JSONs in directory, default locations/categories:
python terra_analysis.py \
    --annotations_dir /path/to/eccv_18_annotation_files \
    --output_index    terra_index.pkl \
    --output_summary  terra_summary.json

# Use specific files only (e.g. exclude test splits):
python terra_analysis.py \
    --annotations_dir  /path/to/eccv_18_annotation_files \
    --include_files    train_annotations.json cis_val_annotations.json \
    --include_locations 38 43 46 \
    --include_categories bobcat squirrel bird raccoon opossum rabbit cat coyote dog empty

Outputs
-------
terra_index.pkl   : serialized index (image_index, loc_cat_to_images,
                    loc_cat_to_seqs, seq_to_images)
terra_summary.json: human-readable per-(location, category) stats
"""

import json
import pickle
import argparse
import glob
import os
from collections import defaultdict
from pathlib import Path


# -- Defaults -----------------------------------------------------------------

DEFAULT_LOCATIONS = {38, 43, 46, 100}
DEFAULT_CATEGORIES = {
    "bird", "bobcat", "cat", "coyote", "dog",
    "empty", "opossum", "rabbit", "raccoon", "squirrel"
}
# Canonical class ordering matching training logs (0-9)
CATEGORY_INDEX = {
    "bird": 0, "bobcat": 1, "cat": 2, "coyote": 3, "dog": 4,
    "empty": 5, "opossum": 6, "rabbit": 7, "raccoon": 8, "squirrel": 9
}
DAY_HOURS = range(6, 20)   # 06:00 - 19:59 inclusive


def cat_label(cat):
    """Return '0/bird' style label for a category name."""
    idx = CATEGORY_INDEX.get(cat, '?')
    return f"{idx}/{cat}"


# -- Helpers ------------------------------------------------------------------

def get_tod(date_str):
    """Return 'day' or 'night' from 'YYYY-MM-DD HH:MM:SS'."""
    hour = int(date_str.split(' ')[1].split(':')[0])
    return 'day' if hour in DAY_HOURS else 'night'


def resolve_annotation_files(annotations_dir, include_files=None):
    """
    Resolve list of annotation JSON paths.
    If include_files is None or empty, glob all *.json in annotations_dir.
    Otherwise use only the specified filenames within annotations_dir.
    Returns list of full paths.
    """
    if include_files:
        paths = [os.path.join(annotations_dir, f) for f in include_files]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Annotation files not found: {missing}")
    else:
        paths = sorted(glob.glob(os.path.join(annotations_dir, '*.json')))
        if not paths:
            raise FileNotFoundError(f"No JSON files found in {annotations_dir}")
    return paths


def load_annotations(annotation_files, verbose=False):
    """
    Load and merge multiple annotation JSON files.
    Returns merged data dict and per-image provenance dict {image_id -> split_name}.
    """
    data = defaultdict(list)
    provenance = {}   # image_id -> source filename (basename)

    for fpath in annotation_files:
        split_name = Path(fpath).name
        if verbose: print(f"  Loading {split_name} ...")
        with open(fpath) as f:
            annots = json.load(f)
        for k, v in annots.items():
            if isinstance(v, list):
                data[k].extend(v)
        # tag images with their source file
        for img in annots.get('images', []):
            provenance[img['id']] = split_name

    return data, provenance


# -- Index builder ------------------------------------------------------------

def build_index(annotations_dir,
                include_files=None,
                include_locations=None,
                include_categories=None,
                verbose=False):
    """
    Build two-way index from annotation files.

    Parameters
    ----------
    annotations_dir     : str       directory containing annotation JSON files
    include_files       : list|None filenames to load; None = all *.json
    include_locations   : set|None  location ints to keep; None = DEFAULT_LOCATIONS
    include_categories  : set|None  category names to keep; None = DEFAULT_CATEGORIES
    verbose             : bool      print progress messages

    Returns
    -------
    image_index : dict
        image_id -> {
            file_name, location, seq_id, frame_num, seq_num_frames,
            date_captured, tod, height, width, split,
            annotations: [{category, bbox, ann_id}]
        }
    loc_cat_to_images : dict
        (location, category) -> [image_id, ...]
    loc_cat_to_seqs : dict
        (location, category) -> [seq_id, ...]
    seq_to_images : dict
        seq_id -> [image_id, ...] sorted by frame_num
    """
    if include_locations is None:
        include_locations = DEFAULT_LOCATIONS
    if include_categories is None:
        include_categories = DEFAULT_CATEGORIES

    annotation_files = resolve_annotation_files(annotations_dir, include_files)
    if verbose:
        print(f"Using {len(annotation_files)} annotation file(s):")
        for p in annotation_files:
            print(f"  {Path(p).name}")

    data, provenance = load_annotations(annotation_files, verbose=verbose)
    category_dict = {item['id']: item['name'] for item in data['categories']}

    if verbose: print("Building image metadata ...")
    image_meta = {}
    for img in data['images']:
        if img['location'] not in include_locations:
            continue
        image_meta[img['id']] = {
            'file_name':       img['file_name'],
            'location':        img['location'],
            'seq_id':          img['seq_id'],
            'frame_num':       img['frame_num'],
            'seq_num_frames':  img['seq_num_frames'],
            'date_captured':   img['date_captured'],
            'tod':             get_tod(img['date_captured']),
            'height':          img['height'],
            'width':           img['width'],
            'split':           provenance.get(img['id'], 'unknown'),
        }

    if verbose: print("Building annotation index ...")
    image_annotations = defaultdict(list)
    for ann in data['annotations']:
        iid = ann['image_id']
        if iid not in image_meta:
            continue
        cat = category_dict.get(ann['category_id'], 'unknown')
        if cat not in include_categories:
            continue
        image_annotations[iid].append({
            'category': cat,
            'bbox':     ann.get('bbox', None),   # [x, y, w, h] or None
            'ann_id':   ann['id'],
        })

    if verbose: print("Building forward index ...")
    image_index = {}
    for iid, meta in image_meta.items():
        image_index[iid] = {**meta, 'annotations': image_annotations.get(iid, [])}

    if verbose: print("Building reverse index ...")
    loc_cat_to_images = defaultdict(list)
    for iid, record in image_index.items():
        for ann in record['annotations']:
            loc_cat_to_images[(record['location'], ann['category'])].append(iid)
    loc_cat_to_images = dict(loc_cat_to_images)

    if verbose: print("Building sequence index ...")
    seq_to_images = defaultdict(list)
    for iid, record in image_index.items():
        seq_to_images[record['seq_id']].append(iid)
    for seq_id in seq_to_images:
        seq_to_images[seq_id].sort(key=lambda iid: image_index[iid]['frame_num'])
    seq_to_images = dict(seq_to_images)

    loc_cat_to_seqs = defaultdict(set)
    for iid, record in image_index.items():
        for ann in record['annotations']:
            loc_cat_to_seqs[(record['location'], ann['category'])].add(record['seq_id'])
    loc_cat_to_seqs = {k: list(v) for k, v in loc_cat_to_seqs.items()}

    return image_index, loc_cat_to_images, loc_cat_to_seqs, seq_to_images


# -- Summary ------------------------------------------------------------------

def build_summary(image_index, loc_cat_to_images, loc_cat_to_seqs, seq_to_images,
                  include_locations=None, include_categories=None):
    """Build summary dict (location, category) -> stats, including split provenance."""
    if include_locations is None:
        include_locations = DEFAULT_LOCATIONS
    if include_categories is None:
        include_categories = DEFAULT_CATEGORIES

    summary = {}
    for loc in sorted(include_locations):
        for cat in sorted(include_categories):
            imgs = loc_cat_to_images.get((loc, cat), [])
            seqs = loc_cat_to_seqs.get((loc, cat), [])
            if not imgs:
                continue
            burst_lens = [len(seq_to_images[s]) for s in seqs]
            avg_burst  = sum(burst_lens) / len(burst_lens)
            day        = sum(1 for iid in imgs if image_index[iid]['tod'] == 'day')
            bbox       = sum(
                1 for iid in imgs
                for ann in image_index[iid]['annotations']
                if ann['category'] == cat and ann['bbox'] is not None
            )
            # provenance: count images per source split
            split_counts = defaultdict(int)
            for iid in imgs:
                split_counts[image_index[iid]['split']] += 1

            summary[f"L{loc}/{cat}"] = {
                'location':     loc,
                'category':     cat,
                'n_images':     len(imgs),
                'n_seqs':       len(seqs),
                'avg_burst':    round(avg_burst, 1),
                'n_day':        day,
                'n_night':      len(imgs) - day,
                'bbox_pct':     round(100 * bbox / len(imgs)),
                'by_split':     dict(split_counts),
            }
    return summary


def print_summary(summary):
    """Print summary table to stdout including per-split provenance."""
    print("=" * 100)
    print(f"{'LOC':<6} {'CATEGORY':<16} {'IMAGES':>7} {'SEQS':>6} "
          f"{'AVG_BURST':>10} {'DAY':>6} {'NIGHT':>6} {'BBOX%':>7}  SPLITS")
    print("=" * 100)
    for key, s in summary.items():
        splits_str = '  '.join(f"{k}:{v}" for k, v in sorted(s['by_split'].items()))
        print(f"L{s['location']:<5} {cat_label(s['category']):<16} {s['n_images']:>7} "
              f"{s['n_seqs']:>6} {s['avg_burst']:>10.1f} "
              f"{s['n_day']:>6} {s['n_night']:>6} {s['bbox_pct']:>6}%  {splits_str}")
    print()


def print_summary_per_species(summary):
    """Print per-species totals across all locations."""
    from collections import defaultdict
    species_stats = defaultdict(lambda: {'n_images': 0, 'n_seqs': 0,
                                          'n_day': 0, 'n_night': 0,
                                          'n_locs': 0})
    for s in summary.values():
        cat = s['category']
        species_stats[cat]['n_images'] += s['n_images']
        species_stats[cat]['n_seqs']   += s['n_seqs']
        species_stats[cat]['n_day']    += s['n_day']
        species_stats[cat]['n_night']  += s['n_night']
        species_stats[cat]['n_locs']   += 1

    print("=" * 65)
    print("PER SPECIES TOTALS")
    print(f"{'CATEGORY':<16} {'IMAGES':>7} {'SEQS':>6} {'LOCS':>6} {'DAY':>6} {'NIGHT':>6}")
    print("=" * 65)
    for cat in sorted(species_stats, key=lambda c: CATEGORY_INDEX.get(c, 99)):
        s = species_stats[cat]
        print(f"{cat_label(cat):<16} {s['n_images']:>7} {s['n_seqs']:>6} "
              f"{s['n_locs']:>6} {s['n_day']:>6} {s['n_night']:>6}")
    print()


def print_summary_per_species_location(summary, include_locations, include_categories):
    """
    Print species x location matrix of image counts in day/night format.
    Each cell shows D/N where D=day images, N=night images.
    TOTAL column shows day/night totals.
    """
    locs = sorted(include_locations)
    cats = sorted(include_categories, key=lambda c: CATEGORY_INDEX.get(c, 99))

    # build lookup: (loc, cat) -> (day, night)
    dn = {}
    for s in summary.values():
        dn[(s['location'], s['category'])] = (s['n_day'], s['n_night'])

    col_w = 20
    header = (f"{'CATEGORY':<16}"
              + ''.join(f"{'L'+str(l):>{col_w}}" for l in locs)
              + f"{'TOTAL':>{col_w}}")
    sep = "=" * len(header)

    print(sep)
    print("IMAGES PER SPECIES x LOCATION  (day/night)")
    print(header)
    print(sep)
    for cat in cats:
        cells = [dn.get((loc, cat), (0, 0)) for loc in locs]
        if sum(d + n for d, n in cells) == 0:
            continue
        row = f"{cat_label(cat):<16}"
        for d, n in cells:
            cell = f"{d}/{n}" if (d + n) > 0 else "-"
            row += f"{cell:>{col_w}}"
        rd = sum(d for d, n in cells)
        rn = sum(n for d, n in cells)
        row += f"{rd}/{rn}".rjust(col_w)
        print(row)

    # totals row
    print("-" * len(header))
    tot_row = f"{'TOTAL':<16}"
    grand_d = grand_n = 0
    for loc in locs:
        cd = sum(dn.get((loc, cat), (0, 0))[0] for cat in cats)
        cn = sum(dn.get((loc, cat), (0, 0))[1] for cat in cats)
        cell = f"{cd}/{cn}" if (cd + cn) > 0 else "-"
        tot_row += f"{cell:>{col_w}}"
        grand_d += cd
        grand_n += cn
    tot_row += f"{grand_d}/{grand_n}".rjust(col_w)
    print(tot_row)
    print()


# -- Serialize / deserialize --------------------------------------------------

def save_index(image_index, loc_cat_to_images, loc_cat_to_seqs, seq_to_images,
               summary, index_path, summary_path, verbose=False):
    """Serialize index to pickle and summary to JSON."""
    if verbose: print(f"Saving index to {index_path} ...")
    with open(index_path, 'wb') as f:
        pickle.dump({
            'image_index':       image_index,
            'loc_cat_to_images': loc_cat_to_images,
            'loc_cat_to_seqs':   loc_cat_to_seqs,
            'seq_to_images':     seq_to_images,
        }, f)

    if verbose: print(f"Saving summary to {summary_path} ...")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose: print("Done.")


def load_index(index_path, verbose=False):
    """Load serialized index from pickle. Returns four dicts."""
    if verbose: print(f"Loading index from {index_path} ...")
    with open(index_path, 'rb') as f:
        d = pickle.load(f)
    return (d['image_index'], d['loc_cat_to_images'],
            d['loc_cat_to_seqs'], d['seq_to_images'])


# -- Main ---------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build TerraInc annotation index')
    parser.add_argument('--adir', default='.',
                        help='Directory containing annotation JSON files')
    parser.add_argument('--jsons',      nargs='*', default=None,
                        help='JSON filenames to load (default: all *.json in dir)')
    parser.add_argument('--locs',  nargs='+', type=int,
                        default=[38, 43, 46],
                        help='Location IDs to include (default: 38 43 46). '
                             'Add 100 explicitly to include the test location.')
    parser.add_argument('--species', nargs='+',
                        default=sorted(DEFAULT_CATEGORIES),
                        help='Category names to include')
    parser.add_argument('--output_index', default='terra_index.pkl',
                        help='Output pickle path for index')
    parser.add_argument('--output_summary',     default='terra_summary.json',
                        help='Output JSON path for human-readable summary')
    parser.add_argument('--debug', action='store_true',
                        help='Print progress messages')
    args = parser.parse_args()

    inc_locs = set(args.locs)
    inc_cats = set(args.species)

    image_index, loc_cat_to_images, loc_cat_to_seqs, seq_to_images = build_index(
        annotations_dir    = args.adir,
        include_files      = args.jsons,
        include_locations  = inc_locs,
        include_categories = inc_cats,
        verbose            = args.debug,
    )

    summary = build_summary(image_index, loc_cat_to_images, loc_cat_to_seqs,
                            seq_to_images, inc_locs, inc_cats)
    print_summary(summary)
    print_summary_per_species(summary)
    print_summary_per_species_location(summary, inc_locs, inc_cats)

    save_index(image_index, loc_cat_to_images, loc_cat_to_seqs, seq_to_images,
               summary, args.output_index, args.output_summary, verbose=args.debug)
