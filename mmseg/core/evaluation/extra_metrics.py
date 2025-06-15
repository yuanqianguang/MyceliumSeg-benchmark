from medpy.metric.binary import hd95, assd
import numpy as np
import cv2

def safe_assd(pred, gt, voxelspacing=None, connectivity=1):
    # 如果pred或gt为空，全0，则无法计算距离
    if (pred.sum() == 0) or (gt.sum() == 0):
        return None  # 或者 return 0, float('inf') 等
    return assd(pred, gt, voxelspacing=voxelspacing, connectivity=connectivity)

def safe_hd95(pred, gt, voxelspacing=None, connectivity=1):
    # 如果预测或标注为空，则返回一个默认值或直接跳过计算
    if (pred.sum() == 0) or (gt.sum() == 0):
        return 0  # 或者 return 0, return float('inf'), etc.
    return hd95(pred, gt, voxelspacing=voxelspacing, connectivity=connectivity)

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2) # img_diag = (4608^2+3456^2)^(1/2) = 5760
    dilation = int(round(dilation_ratio * img_diag))    
    
    # if dilation_ratio is equal to 0.02, the dilation is 115
    # if dilation_ratio is equal to 0.001, the dilation is 5
    
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt.astype(np.uint8), dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou
