from pathlib import Path
import shutil
import random
import re


def _get_base_and_kind(p: Path):
    """
    Return (base, kind) for recognized image patterns, else (None, None).
    kind ∈ {"omniseg", "region", "rand_roi", "uuid", "uuid_patch"}

    - omniseg images: 'im_<index>.tif'/'im_<index>.tiff' → base='im_<index>'
    - region images: '<base>_region<...>.tif/.tiff' → base='<base>' 
    - rand_roi images: '<base>_rand_roi<...>.tif/.tiff' → base='<base>'
    - uuid images: 'im_<uuid>_<string>_5x.tif/.tiff' → base='im_<uuid>_<string>'
    - uuid_patch images: 'im_<uuid>_<string>_5x_patch<N>.tif/.tiff' → base='im_<uuid>_<string>_patch<N>'
    """
    name = p.name
    stem = p.stem  # filename without extension

    # Old-style: im_<index>.tif or.tiff (strict)
    m = re.match(r'^im_(\d+)\.(tif|tiff)$', name)
    if m:
        index = m.group(1)
        return f'im_{index}', 'omniseg'

    # New uuid-patch style: im_<uuid>_<string>_5x_patch<N>.tif/.tiff
    m = re.match(r'^(im_.+_5x_patch\d+)\.(tif|tiff)$', name)
    if m:
        base = m.group(1)
        return base, 'uuid_patch'

    # New uuid-style: im_<uuid>_<string>_5x.tif/.tiff
    m = re.match(r'^(im_.+)_5x\.(tif|tiff)$', name)
    if m:
        base = m.group(1)
        return base, 'uuid'

    if '_region' in stem and '_region_mask' not in stem:
        base = stem.split('_region')[0]
        return base, 'region'

    if '_rand_roi' in stem and '_rand_roi_mask' not in stem:
        base = stem.split('_rand_roi')[0]
        return base, 'rand_roi'

    return None, None


def _is_mask_file(p: Path) -> bool:
    """Recognize mask files across all conventions."""
    stem = p.stem
    return (
        bool(re.match(r'^im_(\d+)_mask', stem)) or
        '_region_mask' in stem or
        '_rand_roi_mask' in stem or
        bool(re.match(r'^im_.+_5x_mask_patch\d+', stem)) or  # New patch masks
        bool(re.match(r'^im_.+_5x_mask', stem))
    )


def _image_base_for_mask(p: Path):
    """
    Given a mask filename, return the expected image base and kind.
    """
    stem = p.stem
    
    # Old omniseg style
    m = re.match(r'^im_(\d+)_mask', stem)
    if m:
        idx = m.group(1)
        return f'im_{idx}', 'omniseg'
    
    # New uuid-patch style mask: im_<uuid>_<string>_5x_mask_patch<N>
    m = re.match(r'^(im_.+)_5x_mask_patch(\d+)', stem)
    if m:
        base_without_5x = m.group(1)
        patch_num = m.group(2)
        base = f'{base_without_5x}_5x_patch{patch_num}'
        return base, 'uuid_patch'
    
    # New uuid-style mask: im_<uuid>_<string>_5x_mask
    m = re.match(r'^(im_.+)_5x_mask', stem)
    if m:
        base = m.group(1)
        return base, 'uuid'
    
    if '_region_mask' in stem:
        base = stem.split('_region_mask')[0]
        return base, 'region'
    if '_rand_roi_mask' in stem:
        base = stem.split('_rand_roi_mask')[0]
        return base, 'rand_roi'
    return None, None


def collect_image_mask_pairs(root_path):
    root = Path(root_path)
    pairs = []

    for folder in root.iterdir():
        if folder.is_dir() and folder.name != 'dataset':
            print(f"\n--- Scanning folder: {folder} ---")

            matched_images = set()
            missing_mask_images = []

            # ====== Collect images and pair with masks ======

            # Old-style images: im_<index>.tif/tiff
            omniseg_style_images = list(folder.glob('im_*.tif')) + list(folder.glob('im_*.tiff'))
            for im_file in omniseg_style_images:
                m = re.match(r'^im_(\d+)\.(tif|tiff)$', im_file.name)
                if not m:
                    continue
                idx = m.group(1)
                base = f'im_{idx}'
                mask_candidates = (
                    list(folder.glob(f'{base}_mask*.tif')) +
                    list(folder.glob(f'{base}_mask*.tiff'))
                )
                if mask_candidates:
                    pairs.append((im_file, mask_candidates[0]))
                    matched_images.add(im_file)
                else:
                    missing_mask_images.append((im_file, base, 'omniseg'))

            # New uuid-patch style images: im_<uuid>_<string>_5x_patch<N>.tif/.tiff
            uuid_patch_images = (
                list(folder.glob('im_*_5x_patch*.tif')) +
                list(folder.glob('im_*_5x_patch*.tiff'))
            )
            for im_file in uuid_patch_images:
                m = re.match(r'^(im_.+)_5x_patch(\d+)\.(tif|tiff)$', im_file.name)
                if not m:
                    continue
                base_without_5x = m.group(1)
                patch_num = m.group(2)
                base = f'{base_without_5x}_5x_patch{patch_num}'
                
                # Look for mask: im_<uuid>_<string>_5x_mask_patch<N>.tiff
                mask_candidates = (
                    list(folder.glob(f'{base_without_5x}_5x_mask_patch{patch_num}.tif')) +
                    list(folder.glob(f'{base_without_5x}_5x_mask_patch{patch_num}.tiff'))
                )
                if mask_candidates:
                    pairs.append((im_file, mask_candidates[0]))
                    matched_images.add(im_file)
                else:
                    missing_mask_images.append((im_file, base, 'uuid_patch'))

            # New uuid-style images: im_<uuid>_<string>_5x.tif/.tiff (non-patch)
            uuid_style_images = (
                list(folder.glob('im_*_5x.tif')) +
                list(folder.glob('im_*_5x.tiff'))
            )
            for im_file in uuid_style_images:
                # Skip if it's a patch file
                if '_5x_patch' in im_file.stem:
                    continue
                    
                m = re.match(r'^(im_.+)_5x\.(tif|tiff)$', im_file.name)
                if not m:
                    continue
                base = m.group(1)
                mask_candidates = (
                    list(folder.glob(f'{base}_5x_mask.tif')) +
                    list(folder.glob(f'{base}_5x_mask.tiff'))
                )
                if mask_candidates:
                    pairs.append((im_file, mask_candidates[0]))
                    matched_images.add(im_file)
                else:
                    missing_mask_images.append((im_file, base, 'uuid'))

            # Region-style images
            rachel_region_images = (
                list(folder.glob('*region*.tif')) +
                list(folder.glob('*region*.tiff'))
            )
            for im_file in rachel_region_images:
                stem = im_file.stem
                if '_region' not in stem or '_region_mask' in stem:
                    continue
                base = stem.split('_region')[0]
                mask_candidates = (
                    list(folder.glob(f'{base}_region_mask*.tif')) +
                    list(folder.glob(f'{base}_region_mask*.tiff'))
                )
                if mask_candidates:
                    pairs.append((im_file, mask_candidates[0]))
                    matched_images.add(im_file)
                else:
                    missing_mask_images.append((im_file, base, 'region'))

            # Rand ROI-style images
            rachel_rand_images = (
                list(folder.glob('*rand_roi*.tif')) +
                list(folder.glob('*rand_roi*.tiff'))
            )
            for im_file in rachel_rand_images:
                stem = im_file.stem
                if '_rand_roi' not in stem or '_rand_roi_mask' in stem:
                    continue
                base = stem.split('_rand_roi')[0]
                mask_candidates = (
                    list(folder.glob(f'{base}_rand_roi_mask*.tif')) +
                    list(folder.glob(f'{base}_rand_roi_mask*.tiff'))
                )
                if mask_candidates:
                    pairs.append((im_file, mask_candidates[0]))
                    matched_images.add(im_file)
                else:
                    missing_mask_images.append((im_file, base, 'rand_roi'))

            # ====== Diagnostics ======

            all_tifs = set(folder.glob('*.tif')) | set(folder.glob('*.tiff'))

            non_category_images = sorted([
                p for p in all_tifs
                if _get_base_and_kind(p)[0] is None and not _is_mask_file(p)
            ], key=lambda x: x.name)

            if non_category_images:
                print("Images NOT matching any category pattern:")
                for p in non_category_images:
                    print(f"  - {p.name}")

            if missing_mask_images:
                print("Category-matching images missing masks (with base):")
                for p, base, kind in sorted(missing_mask_images, key=lambda x: x[0].name):
                    print(f"  - {p.name} | base={base} | kind={kind}")

            masks_without_images = []
            for p in sorted([x for x in all_tifs if _is_mask_file(x)], key=lambda x: x.name):
                base, kind = _image_base_for_mask(p)
                if base is None:
                    masks_without_images.append((p, None, None))
                    continue

                if kind == 'omniseg':
                    possible_images = (
                        list(folder.glob(f'{base}.tif')) +
                        list(folder.glob(f'{base}.tiff'))
                    )
                elif kind == 'uuid_patch':
                    # base is like 'im_<uuid>_<string>_5x_patch<N>'
                    possible_images = (
                        list(folder.glob(f'{base}.tif')) +
                        list(folder.glob(f'{base}.tiff'))
                    )
                elif kind == 'uuid':
                    possible_images = (
                        list(folder.glob(f'{base}_5x.tif')) +
                        list(folder.glob(f'{base}_5x.tiff'))
                    )
                elif kind == 'region':
                    possible_images = (
                        list(folder.glob(f'{base}_region*.tif')) +
                        list(folder.glob(f'{base}_region*.tiff'))
                    )
                elif kind == 'rand_roi':
                    possible_images = (
                        list(folder.glob(f'{base}_rand_roi*.tif')) +
                        list(folder.glob(f'{base}_rand_roi*.tiff'))
                    )
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
    process_and_split("/home/riware/Desktop/mittal_lab/cortex_annotations_formatted_split")