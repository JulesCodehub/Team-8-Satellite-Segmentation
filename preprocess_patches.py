import os
import numpy as np
from tifffile import imread
from skimage.util import view_as_windows
from skimage.transform import resize
from glob import glob
import tifffile

PATCH_SIZE = 128
STRIDE = 16

def make_patches(input_folder, label_folder, output_img_dir, output_mask_dir):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    img_paths = glob(os.path.join(input_folder, '*.tif'))

    patch_count = 0
    for img_path in img_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]
        rgb = imread(img_path)
        mask_path = os.path.join(label_folder, f"{name.replace('RGB', 'CLS')}.tif")
        mask = imread(mask_path)

        # Resize to 128x128
        rgb_resized = resize(rgb, (128, 128), preserve_range=True).astype(np.uint8)
        mask_resized = resize(mask, (128, 128), order=0, preserve_range=True).astype(np.uint8)
        tifffile.imwrite(os.path.join(output_img_dir, f"{name}_resized.tif"), rgb_resized)
        tifffile.imwrite(os.path.join(output_mask_dir, f"{name}_resized.tif"), mask_resized)

        # Slice into 128x128 patches
        if rgb.shape[0] >= PATCH_SIZE and rgb.shape[1] >= PATCH_SIZE:
            rgb_patches = view_as_windows(rgb, (PATCH_SIZE, PATCH_SIZE, 3), step=STRIDE)
            mask_patches = view_as_windows(mask, (PATCH_SIZE, PATCH_SIZE), step=STRIDE)

            for i in range(rgb_patches.shape[0]):
                for j in range(rgb_patches.shape[1]):
                    tifffile.imwrite(os.path.join(output_img_dir, f"{name}_patch_{i}_{j}.tif"),
                                     rgb_patches[i, j, 0])
                    tifffile.imwrite(os.path.join(output_mask_dir, f"{name}_patch_{i}_{j}.tif"),
                                     mask_patches[i, j])

                    patch_count += 1

    print(f"Saved {patch_count} patches.")
