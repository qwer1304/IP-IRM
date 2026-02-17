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
    target_size = args.target_image_size
    input_path = Path(args.in_dir)
    output_path = Path(args.out_dir)

    total_files = count_files(args.in_dir)
    print(f"Begin conversion to {target_size}x{target_size} (Proportional Crop)...")
    
    with tqdm(total=total_files, unit="file", desc="Processing files") as pbar:
        for f in input_path.rglob("*.jpg"):
            # 1. Load (CPU)
            img = cv2.imread(str(f))
            if img is None: 
                continue

            # 2. Upload to GPU
            # Convert HWC BGR to CHW RGB Tensor
            img_t = torch.from_numpy(img).to(device).permute(2, 0, 1).float().unsqueeze(0)

            # 3. Calculate Proportional Resize Dimensions
            # We scale based on the shorter side to avoid squashing
            _, _, h, w = img_t.shape
            scale = target_size / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)

            with torch.no_grad():
                # 4. GPU Proportional Resize (Maintains Aspect Ratio)
                resized_t = T.resize(img_t, (new_h, new_w), interpolation='bilinear')

                # 5. GPU Center Crop to target_size x target_size
                # This ensures the final output is a perfect square (e.g., 224x224)
                final_t = T.center_crop(resized_t, (target_size, target_size))

            # 6. Download to CPU
            final_img = final_t.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()

            # 7. Save Logic
            dest_parent = output_path / f.relative_to(input_path).parent
            dest_parent.mkdir(parents=True, exist_ok=True)

            if args.out_enc == 'JPEG':
                dest = output_path / f.relative_to(input_path)
                cv2.imwrite(str(dest), final_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif args.out_enc == 'WEBP':
                dest = output_path / f.relative_to(input_path).with_suffix('.webp')            
                # Method 0 reduces CPU load during encoding for faster I/O
                cv2.imwrite(str(dest), final_img, [
                    cv2.IMWRITE_WEBP_QUALITY, 80, 
                    cv2.IMWRITE_WEBP_METHOD, 0
                ])            

            pbar.update(1)
        
        print("\nDone!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Global args
    parser.add_argument('--in_dir', type=str, default="./data/DataSets/")
    parser.add_argument('--out_dir', type=str, default="./data/DataSets/")
    parser.add_argument('--target_image_size', type=int, default=224)
    parser.add_argument('--out_enc', type=str, default="JPEG", choices=["WEBP", "JPEG"])
    
    args = parser.parse_args()
    main(args)