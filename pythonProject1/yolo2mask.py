# 将yolo格式的mask标签转化为图像掩码格式的

import os
import cv2
import numpy as np

# YOLO标签的顺序：0-antenna, 1-solar panel, 2-instrument, 3-thruster, 4-opticpayload
id2name = {0: 'antenna', 1: 'solar panel', 2: 'instrument', 3: 'thruster', 4: 'opticpayload'}
class2color = {'antenna': 2, 'solar panel': 1, 'instrument': 3, 'thruster': 4, 'opticpayload': 5}

def yolo_poly_to_points(poly, w, h):
    '''YOLO格式的分割点序列（normalized）转绝对坐标'''
    # 输入格式类似：'0 x1 y1 x2 y2 x3 y3 ...'
    points = []
    for i in range(0, len(poly), 2):
        x = float(poly[i]) * w
        y = float(poly[i + 1]) * h
        points.append([int(x), int(y)])
    return np.array([points], dtype=np.int32)

def txt2mask(txtfile, img_shape):
    # 新建和图片一样大小的mask
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    with open(txtfile, 'r') as f:
        lines = f.readlines()
    for line in lines:
        arr = line.strip().split()
        if len(arr) < 7: continue  # 至少3点才有面
        class_id = int(arr[0])
        name = id2name[class_id]
        color = class2color[name]
        poly = arr[1:]
        points = yolo_poly_to_points(poly, img_shape[1], img_shape[0])
        cv2.fillPoly(mask, points, color)
    return mask

def process(txt_dir, img_dir, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)
    for fname in os.listdir(txt_dir):
        if not fname.endswith('.txt'): continue
        txt_path = os.path.join(txt_dir, fname)
        # 假设图片名和标签名除了扩展名完全相同
        img_name = fname.rsplit('.', 1)[0] + '.png'
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):  # 支持jpg, jpeg
            img_name = fname.rsplit('.', 1)[0] + '.jpg'
            img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            img_name = fname.rsplit('.', 1)[0] + '.jpeg'
            img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Image for {fname} not found, skip.")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image {img_path}, skip.")
            continue
        mask = txt2mask(txt_path, img.shape)
        mask_name = fname.rsplit('.', 1)[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_name)
        cv2.imwrite(mask_path, mask)
        print(f"Saved mask: {mask_path}")



if __name__  == '__main__':
    txt_dir = r'/Volumes/My Passport/dataset/URSO/test/sam_labels_yolo'    # 标签txt所在目录
    img_dir = r'/Volumes/My Passport/dataset/URSO/test/images'   # 原始图片目录
    mask_dir = r'/Volumes/My Passport/dataset/URSO/test/sam_labels_mask'  # 输出mask目录

    process(txt_dir, img_dir, mask_dir)