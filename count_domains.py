import os
import numpy as np
from collections import defaultdict
import argparse


def count_domains(root):
    # discover domains and classes first
    # domains - list
    # classes - set
    # counts - numpy array (domain, class)
    domains = []
    classes = set()

    with os.scandir(root) as domains_iter:
        for d_entry in domains_iter:
            if d_entry.is_dir():
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
            di = d2i[d_entry.name]
            with os.scandir(d_entry.path) as labels_iter:
                for l_entry in labels_iter:
                    if not l_entry.is_dir():
                        continue
                    cj = c2i[l_entry.name]
                    n_images = sum(1 for f in os.scandir(l_entry.path) if f.is_file())
                    counts[di, cj] = n_images
                    
    return domains, classes, counts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Count Classes in Domains')
    parser.add_argument('--root_dir', type=str, default='./')
    
    args = parser.parse_args()

    domains, classes, counts = main(args.root_dir)
    
    # per-class totals (sum across domains, shape = (num_classes,))
    totals_per_class = counts.sum(axis=0)

    print("domains:", domains)
    print("classes:", classes)
    print("counts matrix:\n", counts)
    print("totals per class:", totals_per_class)
