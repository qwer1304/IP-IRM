import os
from PIL import Image
import argparse

def scantree(path):
    """Recursively yield DirEntry objects for given directory."""
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)
        elif entry.is_file():
            yield entry

def main(args):

    os.makedirs(args.out_dir, exist_ok=True) # better safe than sorry

    size = args.target_image_size, args.target_image_size
    for infile in scantree(args.in_dir):

        fnext = infile.name # with file type
        fn, fext = os.path.splitext(fnt) # fext has the '.'
        outfile = os.path.join(args.out_dir, fnext)
      
        if infile != outfile:
            try:
                im = Image.open(infile)
                im.thumbnail(size, Image.Resampling.LANCZOS)
                im.save(outfile, fext[1:]) # use fext as encoding type
            except IOError:
                print("cannot create thumbnail for '%s'" % infile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Global args
    parser.add_argument('--in_dir', type=str, default="./data/DataSets/")
    parser.add_argument('--out_dir', type=str, default="./data/DataSets/")
    parser.add_argument('--target_image_size', type=int, default=224)
    
    args = parser.parse_args()
    
    main(args)
