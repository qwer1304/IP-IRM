import os
import cv2
import torch
import kornia.geometry.transform as T
from pathlib import Path
import argparse
from tqdm import tqdm

def count_files(path):
    total = 0
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            total += count_files(entry.path)
        elif entry.is_file():
            total += 1
    return total

def main(args):

    # Move the resize operation to the GPU
    device = torch.device("cuda")
    size = args.target_image_size
    input_path = Path(args.in_dir)
    output_path = Path(args.out_dir)

    total_files = count_files(args.in_dir)
    print("Begin conversion...")
    with tqdm(total=total_files, unit="file", desc="Processing files") as pbar:
        for f in input_path.rglob("*.jpg"):
            # 1. Load (CPU)
            img = cv2.imread(str(f))
            if img is None: continue

            # 2. Upload to GPU (This is where your CPU gets a break)
            # Convert HWC BGR to CHW RGB Tensor
            img_t = torch.from_numpy(img).to(device).permute(2, 0, 1).float().unsqueeze(0)

            # 3. GPU Resize (Bicubic is the closest high-quality match to Lanczos)
            # This part is nearly instant on GPU
            with torch.no_grad():
                resized_t = T.resize(img_t, (size, size), interpolation='bicubic')

            # 4. Download to CPU
            resized_img = resized_t.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()

            # 5. Save (CPU)
            if args.out_enc == 'JPEG':
                dest = output_path / f.relative_to(input_path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                quality = 95
                cv2.imwrite(str(dest), resized_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif  args.out_enc == 'WEBP':
                dest = output_path / f.relative_to(input_path).with_suffix('.webp')            
                dest.parent.mkdir(parents=True, exist_ok=True)
                # WebP Quality: 80 is usually equivalent to JPEG 95 but much smaller
                quality = 80
                """
                The standard WebP compression (which uses method 4 by default) will be slow. 
                Method 0 tells the encoder: "Do the absolute minimum amount of work to get this into a WebP container." 
                This significantly reduces the CPU load of the imwrite command while still giving the benefit of the WebP format.
                """
                cv2.imwrite(str(dest), resized_img, [cv2.IMWRITE_WEBP_QUALITY, quality, cv2.IMWRITE_WEBP_METHOD, 0])            

            pbar.update(1)
        #end for f in input_path.rglob("*.jpg"):
        print()
        print("Done!")  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Global args
    parser.add_argument('--in_dir', type=str, default="./data/DataSets/")
    parser.add_argument('--out_dir', type=str, default="./data/DataSets/")
    parser.add_argument('--target_image_size', type=int, default=224)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--out_enc', type=str, default="JPEG", choices=["WEBP", "JPEG"])
    
    args = parser.parse_args()
    
    main(args)

