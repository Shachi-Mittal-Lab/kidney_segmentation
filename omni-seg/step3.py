import timeit

import cv2 as cv2
import numpy as np
import os

import numpy as np
import torch
from skimage.transform import resize
import glob
import matplotlib.pyplot as plt

from torchvision import transforms
import torch.multiprocessing as mp
from skimage.transform import resize

from skimage.transform import resize

import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def mapping(img, output_dir, case_name):
    task_list = ['0_1_dt', '1_1_pt', '2_0_capsule', '3_0_tuft', '4_1_vessel', '5_3_ptc']
    upsample_rate = [2, 2, 4, 4, 2, 1]
    colormap = [[1, 0, 0], [-0.5, 1, 0], [-0.5, 0, 1], [1, 1, 0], [1, 0, 1], [-0.5, 1, 1], [1, 1, -0.5]]

    output_folder = os.path.join(output_dir, case_name, os.path.basename(img))
    now_folder = os.path.join(output_folder, task_list[0])

    image = plt.imread(glob.glob(os.path.join(now_folder, 'Big_map.png'))[0])[:, :, :3]
    image = torch.from_numpy(image).cuda()
    image = image / 2
    big_pred = torch.zeros((image.shape)).cuda()
    check_pred = torch.zeros((image.shape)).cuda()

    for ti in range(len(task_list)):
        print(task_list[ti])
        color = colormap[ti]
        now_folder = os.path.join(output_folder, task_list[ti])
        pred = plt.imread(glob.glob(os.path.join(now_folder, 'Big_pred.png'))[0])[:, :, :3]
        pred = torch.from_numpy(pred).cuda()
        pred[pred > 0.5] = 1.
        pred[pred < 1.] = 0.

        check_pred = check_pred + pred
        pred[:, :, 0] = pred[:, :, 0] * color[0]
        pred[:, :, 1] = pred[:, :, 1] * color[1]
        pred[:, :, 2] = pred[:, :, 2] * color[2]

        big_pred = big_pred + pred / 6

    check_pred[check_pred >= 1.] = 1.
    check_pred[check_pred < 1.] = 0.

    check_pred = 1 - check_pred

    big_pred = big_pred + image
    big_pred[big_pred > 1] = 1.
    big_pred[big_pred < 0] = 0.

    check_pred[:, :, 0] = check_pred[:, :, 0] * colormap[-1][0] / 6
    check_pred[:, :, 1] = check_pred[:, :, 1] * colormap[-1][1] / 6
    check_pred[:, :, 2] = check_pred[:, :, 2] * colormap[-1][2] / 6

    check_pred = check_pred + big_pred
    check_pred[check_pred > 1] = 1.
    check_pred[check_pred < 0] = 0.

    big_pred = big_pred.cpu().numpy()
    check_pred = check_pred.cpu().numpy()

    plt.imsave(os.path.join(output_folder, 'merge_pred.png'), big_pred)
    plt.imsave(os.path.join(output_folder, 'other_pred.png'), check_pred)

def slice_merging(big_slice_folder, output_dir, case_name, padding_size):
    task_list = ['0_1_dt', '1_1_pt', '2_0_capsule', '3_0_tuft', '4_1_vessel', '5_3_ptc']
    downsample_rate = 1
    sections = glob.glob(os.path.join(big_slice_folder, case_name + '*'))[0]

    img = plt.imread(sections)[:, :, :3]

    x_length = img.shape[0] + 2 * padding_size
    y_length = img.shape[1] + 2 * padding_size

    pred_image = torch.zeros((x_length, y_length, len(task_list) + 1)).to(torch.uint8).cuda()

    for nt in range(len(task_list)):
        now_task = task_list[nt]
        print(now_task)
        stride_x = int((img.shape[0] + 2 * padding_size) / stride_size) - 1
        stride_y = int((img.shape[1] + 2 * padding_size) / stride_size) - 1
        output_folder = os.path.join(output_dir, case_name)

        for xi in range(stride_x):
            for yi in range(stride_y):
                print(xi, yi)
                x_ind = int(xi * stride_size)
                y_ind = int(yi * stride_size)

                now_size = int(patch_size / downsample_rate)
                now_stride = int(stride_size / downsample_rate)
                x_mid_ind = int(xi * now_stride)
                y_mid_ind = int(yi * now_stride)

                pred_dir = os.path.join(output_folder, '%d_%d' % (x_ind, y_ind), now_task, 'Big_pred.npy')

                if not os.path.exists(pred_dir):
                    now_pred = torch.zeros((patch_size, patch_size)).to(torch.uint8).cuda()  # * 246 / (255 * 2)
                else:
                    now_pred = np.load(pred_dir, allow_pickle=True)[:, :]
                    now_pred = (torch.from_numpy(now_pred)).cuda().to(torch.uint8)

                transform = transforms.Resize(now_size)
                resize_mask = transform(now_pred)
                resize_mask[resize_mask > 0.5] = 1
                resize_mask[resize_mask < 0.5] = 0
                resize_mask = resize_mask.to(torch.uint8)


                boundery = 256


                pred_image[x_mid_ind + boundery: x_mid_ind + now_size - boundery,
                y_mid_ind + boundery: y_mid_ind + now_size - boundery, nt] = torch.add(
                    pred_image[x_mid_ind + boundery: x_mid_ind + now_size - boundery,
                    y_mid_ind + boundery: y_mid_ind + now_size - boundery,
                    nt], resize_mask[boundery:-boundery, boundery:-boundery]).cuda()

    slice_merging_folder = os.path.join(output_folder.replace('segmentation_merge', 'final_merge'))
    image_dir = os.path.join(slice_merging_folder, 'tissue_map_20X.npy')

    contour_map = np.load(image_dir, allow_pickle=True)[:, :, :1]

    contour_map[contour_map > 0.5] = 1
    contour_map[contour_map < 0.5] = 0

    contour_map = (torch.from_numpy(contour_map)).cuda().to(torch.uint8)

    pred_image[padding_size:-padding_size, padding_size:-padding_size, :] = torch.mul(
        pred_image[padding_size:-padding_size, padding_size:-padding_size, :],
        torch.repeat_interleave(contour_map, 7, dim=2).cuda()).cuda()

    row = pred_image.shape[0]
    col = pred_image.shape[1]
    patch_x = 1024
    patch_y = 1024
    for x in range(0, row, patch_x):
        for y in range(0, col, patch_y):

            if row - x < patch_x:
                patch_x = row - x
            elif col - y < patch_y:
                patch_y = col - y

            sum = (torch.sum(pred_image[x:x + patch_x, y:y + patch_y, :len(task_list)], dim=2) == 0).to(torch.uint8).cuda()
            pred_image[x:x + patch_x, y:y + patch_y, len(task_list)] = torch.mul(
                sum, 7).cuda()

    pred_image = pred_image[padding_size:-padding_size, padding_size:-padding_size, :].detach().clone()

    pred_image_tuft_cnt = pred_image[:, :, 3]

    row = pred_image.shape[0]
    col = pred_image.shape[1]

    patch_x = 1024
    patch_y = 1024

    max = torch.zeros(pred_image.shape[0], pred_image.shape[1]).to(torch.uint8).cuda()

    for x in range(0, row, patch_x):
        for y in range(0, col, patch_y):

            if row - x < patch_x:
                patch_x = row - x
            elif col - y < patch_y:
                patch_y = col - y

            max[x:x + patch_x, y:y + patch_y] = torch.argmax(pred_image[x:x + patch_x, y:y + patch_y, :], dim=2)

    pred_image = max

    pred_image_glom_idx = torch.add((pred_image[:, :] == 2), (pred_image[:, :] == 3)).cuda()

    for nt in range(len(task_list)):
        now_task = task_list[nt]
        slice_merging_folder = os.path.join(output_folder.replace('segmentation_merge', 'final_merge'), now_task)

        if nt != 2 and nt != 3:
            now_pred_image = torch.stack(
                [(pred_image[:, :] == nt).to(torch.uint8).cuda(), (pred_image[:, :] == nt).to(torch.uint8).cuda(),
                 (pred_image[:, :] == nt).to(torch.uint8).cuda()], dim=2).cuda()
        elif nt == 2:
            now_pred_image = torch.stack(
                [pred_image_glom_idx.to(torch.uint8).cuda(), pred_image_glom_idx.to(torch.uint8).cuda(),
                 pred_image_glom_idx.to(torch.uint8).cuda()], dim=2)
        else:
            now_pred_image_tuft = torch.mul(pred_image_glom_idx, (pred_image_tuft_cnt >= 2)).cuda()
            now_pred_image = torch.stack(
                [now_pred_image_tuft.to(torch.uint8).cuda(), now_pred_image_tuft.to(torch.uint8).cuda(),
                 now_pred_image_tuft.to(torch.uint8).cuda()], dim=2)

        if not os.path.exists(slice_merging_folder):
            os.makedirs(slice_merging_folder)

        image_dir = os.path.join(slice_merging_folder, 'slice_pred_40X.npy')


        now_pred_image = torch.mul(now_pred_image, 255).cuda()
        now_pred_image = now_pred_image.cpu().numpy().astype(np.uint8)
        np.save(image_dir, now_pred_image, allow_pickle=True)

def filter_contours(contours, hierarchy, filter_params):
    """
        Filter contours by: area.
    """
    filtered = []

    # find indices of foreground contours (parent == -1)
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
    all_holes = []

    # loop through foreground contour indices
    for cont_idx in hierarchy_1:
        # actual contour
        cont = contours[cont_idx]
        # indices of holes contained in this contour (children of parent contour)
        holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
        # take contour area (includes holes)
        a = cv2.contourArea(cont)
        # calculate the contour area of each hole
        hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
        # actual area of foreground contour region
        a = a - np.array(hole_areas).sum()
        if a == 0: continue
        if tuple((filter_params['a_t'],)) < tuple((a,)):
            filtered.append(cont_idx)
            all_holes.append(holes)

    foreground_contours = [contours[cont_idx] for cont_idx in filtered]

    hole_contours = []

    for hole_ids in all_holes:
        unfiltered_holes = [contours[idx] for idx in hole_ids]
        unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
        unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
        filtered_holes = []

        # filter these holes
        for hole in unfilered_holes:
            if cv2.contourArea(hole) > filter_params['a_h']:
                filtered_holes.append(hole)

        hole_contours.append(filtered_holes)

    return foreground_contours, hole_contours

def getContour(big_slice_folder, output_dir, case_name):
    sections = glob.glob(os.path.join(big_slice_folder, case_name + '*' ))[0]

    img = (plt.imread(sections)[:, :, :3] * 255).astype(np.uint8)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space

    mthresh = 7
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)  # Apply median blurring

    sthresh = 20
    sthresh_up = 255

    _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

    close = 4
    kernel = np.ones((close, close), np.uint8)
    img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

    # Find and filter contours
    _, contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # Find contours
    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

    filter_params = {'a_t':20, 'a_h': 16, 'max_n_holes':8}
    foreground_contours, hole_contours = filter_contours(contours, hierarchy, filter_params)  # Necessary for filtering out artifacts

    contours_tissue = foreground_contours
    holes_tissue = hole_contours

    tissue_mask = get_seg_mask(region_size = img.shape, scale = 0, contours_tissue = contours_tissue, holes_tissue = holes_tissue, use_holes=True, offset=(0, 0))
    output_folder = os.path.join(output_dir.replace('segmentation_merge', 'final_merge'), case_name)
    slice_merging_folder = os.path.join(output_folder.replace('segmentation_merge', 'final_merge'))

    if not os.path.exists(slice_merging_folder):
        os.makedirs(slice_merging_folder)

    image_dir = os.path.join(slice_merging_folder, 'tissue_map_20X.png')

    plt.imsave(image_dir, tissue_mask)

def get_seg_mask(region_size, scale, contours_tissue, holes_tissue, use_holes=False, offset=(0, 0)):
    print('\ncomputing foreground tissue mask')
    tissue_mask = np.full(region_size,0).astype(np.uint8)
    offset = tuple((np.array(offset) * np.array(scale) * -1).astype(np.int32))
    contours_holes = holes_tissue
    contours_tissue, contours_holes = zip(
        *sorted(zip(contours_tissue, contours_holes), key=lambda x: cv2.contourArea(x[0]), reverse=True))
    for idx in range(len(contours_tissue)):
        cv2.drawContours(image=tissue_mask, contours=contours_tissue, contourIdx=idx, color=(1,1,1), offset=offset,
                         thickness=-1)

        if use_holes:
            cv2.drawContours(image=tissue_mask, contours=contours_holes[idx], contourIdx=-1, color=(0,0,0),
                             offset=offset, thickness=-1)

    return tissue_mask.astype(np.float32)

if __name__ == "__main__":
    print("starting step 3")
#    map = 0
    slice_map = 1
    slice_map_mapping = 0
    output_dir = './outputs/segmentation_merge'
    big_slice_folder = './input'

    cases = glob.glob(os.path.join(output_dir,'*'))
    cases.sort(key=natural_keys)
    patch_size = 4096
    stride_size = 2048
    padding_size = patch_size - stride_size

    for now_case in cases:
        print(now_case)
        case_name = os.path.basename(now_case)
        images = glob.glob(os.path.join(now_case,'*'))
        images.sort(key=natural_keys)

#        for img in images:
#            if map:
#                mapping(img, output_dir, case_name)

        if slice_map:
            start = timeit.default_timer()
            print(output_dir)
            print(case_name)
            slice_merging(big_slice_folder, output_dir, case_name, padding_size)
            end = timeit.default_timer()
            print('step 3.5 merging:', end - start, 'seconds')
