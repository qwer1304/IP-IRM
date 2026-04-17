import json
from pathlib import Path
import argparse
import torch
import utils
import os

def main(args):
    train_data  = utils.Imagenet(root=args.data + '/train')
    num_domains = len(args.domains)
    num_files = len(train_data)
    envs = torch.zeros(num_files, num_domains)
    memory_hash = utils.compute_dataset_fingerprint(train_data)
    
    with open(args.data + '/' + args.map_file) as f:
        symlinks = json.load(f)

    for index, (path, _) in enumerate(train_data.imgs):    
        # Normalize the path to remove double slashes; keep it as a symlink!
        clean_path = os.path.abspath(os.path.normpath(path))
        target = symlinks.get(clean_path, None)
        if target is None: # shouldn't happen, print debug info
            print()
            print(f"symlinks.json {args.data + '/' + args.map_file}")
            print(f"root {args.data + '/train'}")
            print(f"path {path}")
            print(f"clean path {clean_path}")
            assert False
            
        p = Path(target)
        parts = p.parts
        domain_masks = [pp in args.domains for pp in parts]
        try:
            domain_in_path_idx = domain_masks.index(True)
        except ValueError:
            print(f"path {p} has no domain in domains {args.domains}")
            raise 

        domain = args.domains.index(parts[domain_in_path_idx])
        envs[index][domain] = 1.
    fp = args.data + "/" + "envs_terrainc_train_" + memory_hash[:10]
    fp = os.path.normpath(fp) 
    torch.save({"partitions": [envs]}, fp)
    print(f"envs saved in {fp}")

if __name__ == "__main__":
    # create the top-level parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--map_file', type=str, required=True, help='map file, json')
    parser.add_argument('--domains', type=str, nargs="+", required=True, help='list of domains')
    parser.add_argument('--data', type=str, required=True, help='root of data files')
    
       
    args = parser.parse_args()

    main(args)
