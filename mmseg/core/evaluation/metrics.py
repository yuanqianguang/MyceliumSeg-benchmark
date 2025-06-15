import mmcv
import numpy as np
from mmseg.core.evaluation.extra_metrics import safe_assd, safe_hd95, boundary_iou
from tqdm import tqdm
from PIL import Image

def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray): Prediction segmentation map.
        label (ndarray): Ground truth segmentation map.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')
    # modify if custom classes
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=np.float)
    total_area_union = np.zeros((num_classes, ), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=np.float)
    total_area_label = np.zeros((num_classes, ), dtype=np.float)
    for i in tqdm(range(num_imgs)):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index, label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, \
        total_area_pred_label, total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category IoU, shape (num_classes, ).
    """

    all_acc, acc, iou = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, iou


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category dice, shape (num_classes, ).
    """

    all_acc, acc, dice = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, dice

# ========================================= add assd =========================================
# ========================================= add assd =========================================

def total_average_symmetric_surface_distance(results,
                gt_seg_maps,
                num_classes,
                ignore_index,
                label_map=dict(),
                reduce_zero_label=False):

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    # total_assd = 0
    total_assd = np.zeros((num_classes, ), dtype=np.float)


    for i in tqdm(range(num_imgs)):
        img_assd = img_average_symmetric_surface_distance(results[i], gt_seg_maps[i], num_classes,
                                ignore_index, label_map, reduce_zero_label)
        total_assd += np.full((num_classes,), img_assd, dtype=np.float)

    return total_assd / num_imgs

def img_average_symmetric_surface_distance(pred_label,
                                            label,
                                            num_classes,
                                            ignore_index,
                                            label_map=dict(),
                                            reduce_zero_label=False):

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')

    h, w = pred_label.shape

    # modify if custom classes
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    # Reshape to original image size
    pred_label = pred_label.reshape((h, w))
    label = label.reshape((h, w))

    img_assd = safe_assd(pred_label, label, connectivity=1)

    return img_assd

# ========================================= add hd95 =========================================
# ========================================= add hd95 =========================================

def total_hausdorff_distance_95(results,
                                gt_seg_maps,
                                num_classes,
                                ignore_index,
                                label_map=dict(),
                                reduce_zero_label=False):

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    # total_hd95 = 0
    total_hd95 = np.zeros((num_classes, ), dtype=np.float)

    for i in tqdm(range(num_imgs)):
        img_hd95 = img_hausdorff_distance_95(results[i], gt_seg_maps[i], num_classes,
                                ignore_index, label_map, reduce_zero_label)
        total_hd95 += np.full((num_classes,), img_hd95, dtype=np.float)

    return total_hd95 / num_imgs

def img_hausdorff_distance_95(pred_label,
                            label,
                            num_classes,
                            ignore_index,
                            label_map=dict(),
                            reduce_zero_label=False):

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')

    h, w = pred_label.shape

    # modify if custom classes
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    # Reshape to original image size
    pred_label = pred_label.reshape((h, w))
    label = label.reshape((h, w))

    img_hd95 = safe_hd95(pred_label, label, connectivity=1)

    return img_hd95

# ========================================= add biou =========================================
# ========================================= add biou =========================================

def total_boundary_iou(results,
                        gt_seg_maps,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False,
                        dilation_ratio=0.02):

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    # total_biou = 0
    total_biou = np.zeros((num_classes, ), dtype=np.float)

    for i in tqdm(range(num_imgs)):
        img_biou = img_boundary_iou(results[i], gt_seg_maps[i], num_classes,
                                ignore_index, label_map, reduce_zero_label, dilation_ratio)
        total_biou += np.full((num_classes,), img_biou, dtype=np.float)

    return total_biou / num_imgs

def img_boundary_iou(pred_label,
                    label,
                    num_classes,
                    ignore_index,
                    label_map=dict(),
                    reduce_zero_label=False,
                    dilation_ratio=0.02):

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')

    h, w = pred_label.shape

    # modify if custom classes
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    # Reshape to original image size
    pred_label = pred_label.reshape((h, w))
    label = label.reshape((h, w))

    img_biou = boundary_iou(label, pred_label, dilation_ratio)

    return img_biou


# ========================================= eval metric =========================================
# ========================================= eval metric =========================================










def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category evalution metrics, shape (num_classes, ).
    """
    
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'assd', 'hd95', 'biou']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(results, gt_seg_maps,
                                                     num_classes, ignore_index,
                                                     label_map,
                                                     reduce_zero_label)

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    # valid_classes = total_area_label > 0
    # acc = np.zeros_like(total_area_intersect)
    # acc[valid_classes] = total_area_intersect[valid_classes] / total_area_label[valid_classes]
    acc = total_area_intersect / total_area_label
    ret_metrics = [all_acc, acc]
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            ret_metrics.append(iou)
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            ret_metrics.append(dice)
        elif metric == 'assd':
            assd = total_average_symmetric_surface_distance(results, gt_seg_maps, 
                                                num_classes, ignore_index, 
                                                label_map, reduce_zero_label)
            ret_metrics.append(assd)
            # ret_metrics.append(np.full((num_classes,), 0, dtype=np.float))
        elif metric == 'hd95':
            hd95 = total_hausdorff_distance_95(results, gt_seg_maps, 
                                                num_classes, ignore_index, 
                                                label_map, reduce_zero_label)
            ret_metrics.append(hd95)
            # ret_metrics.append(np.full((num_classes,), 0, dtype=np.float))
        elif metric == 'biou':
            biou = total_boundary_iou(results, gt_seg_maps, 
                                        num_classes, ignore_index, 
                                        label_map, reduce_zero_label, dilation_ratio=0.001)
            ret_metrics.append(biou)
            # ret_metrics.append(np.full((num_classes,), 0, dtype=np.float))
            
            
            
    if nan_to_num is not None:
        ret_metrics = [
            np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics
        ]
    return ret_metrics
