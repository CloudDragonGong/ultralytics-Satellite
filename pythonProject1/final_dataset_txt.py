import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


def remove_mask_suffix(filename):
    return filename.replace('_mask', '')


def is_image_file(filename):
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
    return filename.endswith(image_extensions)


def is_valid_image_file(filename):
    return is_image_file(filename) and not filename.startswith('._')


def convert_mask_images_to_yolo_txt(image_folder_path, txt_folder_path):
    if (not os.path.exists(txt_folder_path)) or (not os.path.isdir(txt_folder_path)):
        raise ValueError("The folder path is not valid")
    files = [f for f in os.listdir(image_folder_path) if is_valid_image_file(f)]
    for mask_filename in tqdm(files, desc="Converting Mask Images to YOLO TXT"):
        convert_single_mask_image_to_yolo_txt(image_folder_path, txt_folder_path, mask_filename)


def convert_single_mask_image_to_yolo_txt(image_folder_path, txt_folder_path, mask_filename):
    if os.path.exists(os.path.join(txt_folder_path, os.path.splitext(remove_mask_suffix(mask_filename))[0] + '.txt')):
        # print("{} already converted".format(mask_filename))
        os.remove(os.path.join(txt_folder_path, os.path.splitext(remove_mask_suffix(mask_filename))[0] + '.txt'))
    # 读取掩码图像
    image_path = os.path.join(image_folder_path, mask_filename)
    mask_image = cv2.imread(str(image_path))  # 读取图像
    if mask_image is None:
        print("No mask image for {}".format(mask_filename))
        return
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
    mask_image_width = mask_image.shape[1]
    mask_image_height = mask_image.shape[0]
    color_to_class = {
        (0, 0, 255): 0,  # blue ->  antenna
        (0, 255, 0): 1,  # green ->  body
        (255, 0, 0): 2  # red -> solar panel
    }
    contours_dict = {0: [], 1: [], 2: []}

    # 获取类别的边缘点
    for color, class_index in color_to_class.items():
        # 创建掩码
        mask = np.all(mask_image == color, axis=-1).astype(np.uint8) * 255

        # 查找边缘并提取轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 将轮廓转换为平面点
            points = contour.reshape(-1, 2)
            normalized_points = points / np.array([mask_image_width, mask_image_height])
            normalized_points = normalized_points.reshape(-1)
            if len(normalized_points) < 3:
                continue
            contours_dict[class_index].append(normalized_points)
        # 保存为 YOLO 格式的 TXT 文件
    output_txt_path = os.path.join(txt_folder_path,
                                   os.path.splitext(remove_mask_suffix(mask_filename))[0] + '.txt')  # 输出文件路径
    with open(output_txt_path, 'w') as f:
        for class_index, points_list in contours_dict.items():
            for points in points_list:
                # 创建格式字符串
                points_str = ' '.join(map(str, points))
                f.write(f"{class_index} {points_str}\n")

    # print(f"Contours saved to {output_txt_path}")


if __name__ == "__main__":
    mask_folder_path = '/Volumes/My Passport/dataset/Final_dataset largest/mask/val'
    label_folder_path = '/Volumes/My Passport/dataset/Final_dataset largest/yolov11/val/labels'
    convert_mask_images_to_yolo_txt(mask_folder_path, label_folder_path)
