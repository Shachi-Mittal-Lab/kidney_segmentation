from pathlib import Path
import shutil
import random
import re

def collect_image_mask_pairs(root_path):
    root = Path(root_path)
    pairs = []

    for folder in root.iterdir():
        if folder.is_dir() and folder.name != 'dataset':
            # Collect old-style files: im_<index>.tif
            omniseg_style_images = list(folder.glob('im_*.tif'))
            for im_file in omniseg_style_images:
                match = re.match(r'im_(\d+)\.tif', im_file.name)
                if match:
                    index = match.group(1)
                    mask_candidates = list(folder.glob(f'im_{index}_mask*.tif'))
                    if mask_candidates:
                        pairs.append((im_file, mask_candidates[0]))

            # Collect new-style files: *_rand_roi.tif
            rachel_style_images = list(folder.glob('*_rand_roi.tif'))
            for im_file in rachel_style_images:
                base = im_file.stem.replace('_rand_roi', '')
                mask_candidates = list(folder.glob(f'{base}_rand_roi_mask_*.tif'))
                if mask_candidates:
                    pairs.append((im_file, mask_candidates[0]))

    return pairs

def save_pairs(pairs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, (im_file, mask_file) in enumerate(pairs):
        pair_dir = output_dir / str(idx)
        pair_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(im_file, pair_dir / 'image.tif')
        shutil.copy(mask_file, pair_dir / 'mask.tif')

def process_and_split(root_path, train_ratio=0.9):
    root = Path(root_path)
    dataset_dir = root / 'dataset'
    training_dir = dataset_dir / 'training'
    validation_dir = dataset_dir / 'validation'

    pairs = collect_image_mask_pairs(root)

    random.shuffle(pairs)
    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    save_pairs(train_pairs, training_dir)
    save_pairs(val_pairs, validation_dir)

if __name__ == "__main__":
    process_and_split("/mnt/Data/kidney/rachel_training_data_10x/glom/carter_rachel_glom_annotations")