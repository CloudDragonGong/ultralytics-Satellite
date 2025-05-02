import os
import numpy as np
from PIL import Image



class2color = {
    'antenna': 2,
    'solar panel': 1,
    'instrument': 3,
    'thruster': 4,
    'opticpayload': 5
}

def get_class_name(class_id):
    # 通过值反向查找对应的键
    for name, id_ in class2color.items():
        if id_ == class_id:
            return name
    return None  # 如果没有匹配返回None




def compute_iou_per_class(pred, label, class_id):
    pred_mask = (pred == class_id)
    label_mask = (label == class_id)
    intersection = np.logical_and(pred_mask, label_mask).sum()
    union = np.logical_or(pred_mask, label_mask).sum()
    if union == 0:
        return np.nan  # 忽略未出现的类别
    return intersection / union

def calculate_miou(gt_dir, pred_dir, class_ids):
    iou_dict = {cls: [] for cls in class_ids}
    file_names = [f for f in os.listdir(gt_dir) if f.endswith('.png')]
    for fname in file_names:
        gt_path = os.path.join(gt_dir, fname)
        pred_path = os.path.join(pred_dir, fname)
        if not os.path.exists(pred_path):
            print(f"Warning: {fname} not in pred_dir, skip.")
            continue
        gt = np.array(Image.open(gt_path))
        pred = np.array(Image.open(pred_path))
        if gt.shape != pred.shape:
            print(f"Warning: {fname} shape mismatch, skip.")
            continue
        for cls in class_ids:
            iou = compute_iou_per_class(pred, gt, cls)
            if not np.isnan(iou):
                iou_dict[cls].append(iou)
    miou_dict = {}
    for cls in class_ids:
        cls_ious = iou_dict[cls]
        miou_dict[cls] = np.mean(cls_ious) if len(cls_ious) > 0 else float('nan')
    overall_miou = np.nanmean(list(miou_dict.values()))
    return miou_dict, overall_miou

if __name__ == "__main__":
    # -----------用户自定义部分-------------
    gt_dir = r'/Volumes/My Passport/dataset/UESD/val-1'  # 真实mask标签文件夹路径
    pred_dir = r'/Volumes/My Passport/dataset/UESD/yolov11/val/yolo-seg-mask-pred'  # 待验证mask标签文件夹路径
    # 只包含你的类别，背景可以不算（如果0是背景，可以不计入）
    class_ids = [1, 2, 3, 4, 5]  # 根据你的mask class2color配置
    # --------------------------------
    miou_per_class, overall_miou = calculate_miou(gt_dir, pred_dir, class_ids)
    print("每个类别的mIoU:")
    for cls in class_ids:
        print(f"类别{get_class_name(cls)}: mIoU = {miou_per_class[cls]:.4f}")
    print(f"总的 mean IoU: {overall_miou:.4f}")