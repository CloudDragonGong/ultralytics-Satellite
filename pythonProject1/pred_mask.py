import os

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm


def remove_mask_suffix(filename):
    return filename.replace('_mask', '')


def is_image_file(filename):
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
    return filename.endswith(image_extensions)


def is_valid_image_file(filename):
    return is_image_file(filename) and not filename.startswith('._')




# 类别与灰度值映射
class2color = {
    'antenna': 2,
    'solar panel': 1,
    'instrument': 3,
    'thruster': 4,
    'opticpayload': 5
}
# 你的模型的类别ID与名称，需要和训练集一致
id2name = {0: 'antenna', 1: 'solar panel', 2: 'instrument', 3: 'thruster', 4: 'opticpayload'}

def mask_from_yoloout(masks, classes, id2name, class2color, shape):
    out_mask = np.zeros(shape, dtype=np.uint8)
    for i, mask in enumerate(masks):
        class_id = classes[i]
        name = id2name.get(class_id)
        color = class2color.get(name, 0)  # 找不到的话用0
        # mask shape: (h, w), bool or 0/1
        mask_resized = cv2.resize(mask.astype('float32'),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)
        out_mask[mask_resized.astype(bool)] = color
    return out_mask

def run_segmentation_predict(
    model_path,
    img_dir,
    save_dir,
    id2name,
    class2color
):
    os.makedirs(save_dir, exist_ok=True)
    # 加载模型
    model = YOLO(model_path)
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_name in tqdm(img_files):
        if not is_valid_image_file(img_name):
            continue
        img_path = os.path.join(img_dir, img_name)
        # 推理
        results = model(img_path)
        result = results[0]
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []
        classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else []

        if result.orig_shape is not None:
            h, w = result.orig_shape
        else:  # fallback
            with Image.open(img_path) as im:
                w, h = im.size

        # 如果没有分割掩码，则全0
        if len(masks) == 0:
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            # 合成一张单通道灰度mask
            mask = mask_from_yoloout(masks, classes, id2name, class2color, (h, w))
        # 保存
        mask_save_name = os.path.splitext(img_name)[0] + '.png'
        mask_save_path = os.path.join(save_dir, mask_save_name)
        Image.fromarray(mask).save(mask_save_path)
        # print(f"Saved: {mask_save_path}")

model_path = '/Users/mac/data/workspace/ultralytics-Satellite/runs/segment/train14/weights/best.pt'
img_dir = '/Volumes/My Passport/dataset/UESD/yolov11/val/images'
save_dir ='/Volumes/My Passport/dataset/UESD/yolov11/val/yolo-seg-mask-pred'

if __name__ == '__main__':
    run_segmentation_predict(
        model_path,
        img_dir,
        save_dir,
        id2name,
        class2color
    )