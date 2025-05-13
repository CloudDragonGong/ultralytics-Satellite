from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
def count_parts(image_path):
    # 定义部件与灰度值的映射
    part_mapping = {
        0: 'background',
        1: 'panel',
        2: 'antenna',
        3: 'instrument',
        4: 'thruster',
        5: 'opticpayload'
    }

    # 打开灰度图像
    image = Image.open(image_path).convert('L')  # 确保图像为灰度模式
    image_array = np.array(image)

    # 初始化计数字典
    part_counts = {part: 0 for part in part_mapping.values()}

    # 统计每个部件的像素个数
    for gray_value, part in part_mapping.items():
        part_counts[part] = np.sum(image_array == gray_value)

    return part_counts

def remove_mask_suffix(filename):
    return filename.replace('_mask', '')


def is_image_file(filename):
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
    return filename.endswith(image_extensions)


def is_valid_image_file(filename):
    return is_image_file(filename) and not filename.startswith('._')



# 定义每个类别的 RGB 颜色
color_mapping = {
    0: (0, 0, 0),         # background: 黑色
    1: (255, 0, 0),       # panel: 红色
    2: (0, 255, 0),       # antenna: 绿色
    3: (0, 0, 255),       # instrument: 蓝色
    4: (255, 255, 0),     # thruster: 黄色
    5: (255, 165, 0)      # opticpayload: 橙色
}

def convert_to_rgb(grayscale_image_path):
    # 读取灰度图像并转换为 Array
    image = Image.open(grayscale_image_path).convert('L')  # 确保图像为灰度模式
    image_array = np.array(image)

    # 创建 RGB 图像
    height, width = image_array.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 根据每个像素的灰度值填充相应的 RGB 颜色
    for gray_value, color in color_mapping.items():
        rgb_image[image_array == gray_value] = color

    return rgb_image


def convert_masks_to_rgb(dir_path,rgb_path):
    files = [f for f in os.listdir(dir_path) if is_valid_image_file(f)]
    for filename in tqdm(files):
        imageRGB = convert_to_rgb(os.path.join(dir_path,filename))
        mask_path = os.path.join(rgb_path,filename)
        cv2.imwrite(mask_path,imageRGB)



if __name__ == "__main__":
    mask_path = r'/Volumes/My Passport/dataset/UESD/test/truth_labels'
    rgb_path = r'/Volumes/My Passport/dataset/UESD/test/rgb_labels'
    convert_masks_to_rgb(mask_path,rgb_path)
