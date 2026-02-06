import json
from pathlib import Path
import argparse
import torch
import utils

def main(args):
    train_data  = utils.Imagenet(root=args.data + '/train')
    num_domains = len(args.domains)
    num_files = len(train_data)
    envs = torch.zeros(num_files, num_domains)
    
    with open(args.data + '/' + args.map_file) as f:
        symlinks = json.load(f)

    for index, (path, _) in enumerate(train_data.imgs):
        target = symlinks[path]
        p = Path(target)
        parts = p.parts
        domain_name = parts[9]
        try:
            domain = args.domains.index(domain_name)
        except ValueError:
            print(f"domain {domain_name} not in domains {args.domains}")
            raise 

        envs[index][domain] = 1.
        
    torch.save(list(envs), args.data + "/" + "envs_terrainc_train")
    print(f"envs saved in {args.data + '/' + 'envs_terrainc_train'}")

if __name__ == "__main__":
    # create the top-level parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--map_file', type=str, required=True, help='map file, json')
    parser.add_argument('--domains', type=str, nargs="+", required=True, help='list of domains')
    parser.add_argument('--data', type=str, required=True, help='root of data files')
    
       
    args = parser.parse_args()

    main(args)
