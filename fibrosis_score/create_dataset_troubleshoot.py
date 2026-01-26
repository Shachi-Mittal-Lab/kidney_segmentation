
from pathlib import Path
import shutil
import random
import re


def _get_base_and_kind(p: Path):
    """
    Return (base, kind) for recognized image patterns, else (None, None).
    kind ∈ {"omniseg", "region", "rand_roi"}

    - omniseg images: 'im_<index>.tif' → base='im_<index>'
    - region images: '<base>_region<...>.tif' → base='<base>' (substring before '_region')
    - rand_roi images: '<base>_rand_roi<...>.tif' → base='<base>' (substring before '_rand_roi')
    """
    name = p.name
    stem = p.stem  # filename without extension

    # Old-style: im_<index>.tif (strict)
    m = re.match(r'^im_(\d+)\.tif$', name)
    if m:
        index = m.group(1)
        return f'im_{index}', 'omniseg'

    # Broader: contains '_region' token anywhere in stem (excluding masks)
    if '_region' in stem and '_region_mask' not in stem:
        base = stem.split('_region')[0]  # part before the token
        return base, 'region'

    # Broader: contains '_rand_roi' token anywhere in stem (excluding masks)
    if '_rand_roi' in stem and '_rand_roi_mask' not in stem:
        base = stem.split('_rand_roi')[0]  # part before the token
        return base, 'rand_roi'

    return None, None


def _is_mask_file(p: Path) -> bool:
    """Recognize mask files across the three conventions."""
    stem = p.stem
    return (
        bool(re.match(r'^im_(\d+)_mask', stem)) or
        '_region_mask' in stem or
        '_rand_roi_mask' in stem
    )


def _image_base_for_mask(p: Path):
    """
    Given a mask filename, return the expected image base and kind.
    Returns (base, kind) or (None, None) if not recognized.
    """
    stem = p.stem
    m = re.match(r'^im_(\d+)_mask', stem)
    if m:
        idx = m.group(1)
        return f'im_{idx}', 'omniseg'
    if '_region_mask' in stem:
        base = stem.split('_region_mask')[0]
        return base, 'region'
    if '_rand_roi_mask' in stem:
        base = stem.split('_rand_roi_mask')[0]
        return base, 'rand_roi'
    return None, None


def collect_image_mask_pairs(root_path):
    """
    Scans subfolders of root_path (excluding 'dataset') and collects image–mask pairs.
    Also prints diagnostics:
      - Images not matching any category (excluding mask files)
      - Category-matching images missing masks (with base + kind)
      - Masks without corresponding images (with base + kind)
    """
    root = Path(root_path)
    pairs = []

    for folder in root.iterdir():
        if folder.is_dir() and folder.name != 'dataset':
            print(f"\n--- Scanning folder: {folder} ---")

            matched_images = set()
            missing_mask_images = []  # tuples: (image_path, base, kind)

            # ====== Collect images and pair with masks ======

            # Old-style images: im_<index>.tif (strict)
            omniseg_style_images = list(folder.glob('im_*.tif'))
            for im_file in omniseg_style_images:
                m = re.match(r'^im_(\d+)\.tif$', im_file.name)
                if not m:
                    continue
                idx = m.group(1)
                base = f'im_{idx}'
                mask_candidates = list(folder.glob(f'{base}_mask*.tif'))
                if mask_candidates:
                    pairs.append((im_file, mask_candidates[0]))
                    matched_images.add(im_file)
                else:
                    missing_mask_images.append((im_file, base, 'omniseg'))

            # Region-style images (BROADER): *region*.tif (but must contain '_region' token)
            rachel_region_images = list(folder.glob('*region*.tif'))
            for im_file in rachel_region_images:
                stem = im_file.stem
                if '_region' not in stem or '_region_mask' in stem:
                    continue  # skip non-token or masks
                base = stem.split('_region')[0]
                mask_candidates = list(folder.glob(f'{base}_region_mask*.tif'))
                if mask_candidates:
                    pairs.append((im_file, mask_candidates[0]))
                    matched_images.add(im_file)
                else:
                    missing_mask_images.append((im_file, base, 'region'))

            # Rand ROI-style images (BROADER): *rand_roi*.tif (must contain '_rand_roi' token)
            rachel_rand_images = list(folder.glob('*rand_roi*.tif'))
            for im_file in rachel_rand_images:
                stem = im_file.stem
                if '_rand_roi' not in stem or '_rand_roi_mask' in stem:
                    continue  # skip non-token or masks
                base = stem.split('_rand_roi')[0]
                mask_candidates = list(folder.glob(f'{base}_rand_roi_mask*.tif'))
                if mask_candidates:
                    pairs.append((im_file, mask_candidates[0]))
                    matched_images.add(im_file)
                else:
                    missing_mask_images.append((im_file, base, 'rand_roi'))

            # ====== Diagnostics ======

            # All .tif files in folder
            all_tifs = set(folder.glob('*.tif'))

            # 1) Files that do not fall into any recognized image category (exclude masks)
            non_category_images = sorted([
                p for p in all_tifs
                if _get_base_and_kind(p)[0] is None and not _is_mask_file(p)
            ], key=lambda x: x.name)

            if non_category_images:
                print("Images NOT matching any category pattern:")
                for p in non_category_images:
                    print(f"  - {p.name}")

            # 2) Images that match a category but have NO corresponding mask (print base)
            if missing_mask_images:
                print("Category-matching images missing masks (with base):")
                for p, base, kind in sorted(missing_mask_images, key=lambda x: x[0].name):
                    print(f"  - {p.name} | base={base} | kind={kind}")

            # 3) Masks that have NO corresponding image (broad matching for presence)
            masks_without_images = []
            for p in sorted([x for x in all_tifs if _is_mask_file(x)], key=lambda x: x.name):
                base, kind = _image_base_for_mask(p)
                if base is None:
                    masks_without_images.append((p, None, None))
                    continue

                # Broadly check: images can have extra suffixes after token
                if kind == 'omniseg':
                    possible_images = list(folder.glob(f'{base}.tif'))
                elif kind == 'region':
                    possible_images = list(folder.glob(f'{base}_region*.tif'))
                elif kind == 'rand_roi':
                    possible_images = list(folder.glob(f'{base}_rand_roi*.tif'))
                else:
                    possible_images = []

                if not possible_images:
                    masks_without_images.append((p, base, kind))

            if masks_without_images:
                print("Mask files with NO corresponding image (with base):")
                for p, base, kind in masks_without_images:
                    if base:
                        print(f"  - {p.name} | base={base} | kind={kind}")
                    else:
                        print(f"  - {p.name} | base=<unknown> | kind=<unknown>")

    return pairs


def save_pairs(pairs, output_dir: Path):
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

    pairs = collect_image_mask_pairs(root_path)

    random.shuffle(pairs)
    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    save_pairs(train_pairs, training_dir)
    save_pairs(val_pairs, validation_dir)


if __name__ == "__main__":
    process_and_split("/home/riware/Desktop/mittal_lab/tubule_annotations/dt_pt_w_UIC_formatted")
