import os
from PIL import Image
import argparse


from pathlib import Path
from PIL import Image
import os

src = Path("/kaggle/working/dataset_jpg")   # your current dataset
dst = Path("/kaggle/working/dataset_webp")  # output
dst.mkdir(parents=True, exist_ok=True)

for path in src.rglob("*.jpg"):
    rel = path.relative_to(src)
    outpath = dst / rel.with_suffix(".webp")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if not outpath.exists():  # skip if already converted
        Image.open(path).save(outpath, "WEBP", quality=95)  # lossless if quality=100



def scantree(path, progress=False, level=0):
    """Recursively yield DirEntry objects for given directory."""
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            if progress:
                print('\t'*level + f'Entering {entry.path} ...')
            yield from scantree(entry.path, progress=progress, level=level+1)
            if progress:
                print('\t'*level + f'Done.')
        elif entry.is_file():
            yield entry
def path_from_depth(path, depth):
    parts = path.strip(os.sep).split(os.sep)
    return os.sep.join(parts[-(depth+1):])
    
def main(args):

    size = args.target_image_size
    print("Begin conversion...")
    for infile in scantree(args.in_dir, progress=args.progress, level=0):

        fnext = infile.name # with file ext
        fn, fext = os.path.splitext(fnext) # fext has the '.'
        relfnext = path_from_depth(infile.path, args.depth)
        outfile = os.path.join(args.out_dir, relfnext)
        if args.out_enc == "WEBP":
            outfile = os.path.splitext(outfile)[0] + ".webp"
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
            if args.out_enc == "WEBP":
                im.save(outfile, "WEBP", lossless=True, subsampling=0, optimize=True)
            else:
                im.save(outfile, "JPEG", quality=95, subsampling=0, optimize=True) # overwrites file if it exists

    print("Done!")  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Global args
    parser.add_argument('--in_dir', type=str, default="./data/DataSets/")
    parser.add_argument('--out_dir', type=str, default="./data/DataSets/")
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--target_image_size', type=int, default=224)
    parser.add_argument('--out_enc', type=str, default="WEBP", choices=["WEBP", "JPEG"])
    parser.add_argument('--progress', action='store_true')
    
    args = parser.parse_args()
    
    main(args)
