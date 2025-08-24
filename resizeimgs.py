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
def path_from_depth(path, depth):
    parts = path.strip(os.sep).split(os.sep)
    return os.sep.join(parts[-(depth+1):])
    
def main(args):

    size = args.target_image_size
    print("Begin conversion...")
    for infile in scantree(args.in_dir):

        fnext = infile.name # with file ext
        fn, fext = os.path.splitext(fnext) # fext has the '.'
        relfnext = path_from_depth(infile.path, args.depth)
        outfile = os.path.join(args.out_dir, relfnext)
        outpath = os.path.dirname(outfile)
        os.makedirs(outpath, exist_ok=True) # better safe than sorry

        #print(fnext, outfile)
      
        if (infile.path != outfile):
            try:
                im = Image.open(infile.path)
            except IOError:
                print("cannot open '%s'" % infile.path)
            try:
                w, h = im.size
                scale = size / max(w, h)
                new_size = (int(round(w * scale)), int(round(h * scale)))
                im.resize(new_size, Image.Resampling.LANCZOS)
            except IOError:
                print(f"cannot resize {infile.path} to ({w},{h})")
            try:
                enc = fext[1:]
                if enc.upper() == 'JPG':
                    enc = 'JPEG'
                im.save(outfile, format=enc, quality=95, subsampling=0, optimize=True) # use fext as encoding type
            except IOError:
                print(f"cannot save thumbnail for {infile.path} into {outfile}")
    print("Done!")  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Global args
    parser.add_argument('--in_dir', type=str, default="./data/DataSets/")
    parser.add_argument('--out_dir', type=str, default="./data/DataSets/")
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--target_image_size', type=int, default=224)
    
    args = parser.parse_args()
    
    main(args)
