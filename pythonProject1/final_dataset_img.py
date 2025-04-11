import os
import shutil

from tqdm import tqdm


def remove_mask_suffix(filename):
    return filename.replace('_mask', '')


def is_image_file(filename):
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
    return filename.endswith(image_extensions)


def is_valid_image_file(filename):
    return is_image_file(filename) and not filename.startswith('._')


def copy_matching_images(txt_folder, image_folder, destination_folder):
    # 确保目标文件夹存在，如果不存在则创建它
    os.makedirs(destination_folder, exist_ok=True)

    # 获取所有的 txt 文件名（不带扩展名）
    txt_files = [os.path.splitext(filename)[0] for filename in os.listdir(txt_folder) if filename.endswith('.txt')]

    image_files = [f for f in os.listdir(image_folder) if is_valid_image_file(f)]
    # 遍历图片文件夹中的所有文件
    for image_filename in tqdm(image_files):
        # 获取没有扩展名的图片文件名
        image_name, ext = os.path.splitext(image_filename)

        # 检查图像文件名是否在 txt 文件名列表中
        if image_name in txt_files:
            # 构建完整的源路径和目标路径
            src_image_path = os.path.join(image_folder, image_filename)
            dest_image_path = os.path.join(destination_folder, image_filename)

            # 复制图像文件到目标文件夹
            shutil.copy2(src_image_path, dest_image_path)
            # print(f"复制文件: {src_image_path} 到 {dest_image_path}")
        else:
            print(image_name)
        # 使用示例


if __name__ == '__main__':
    txt_folder = '/Volumes/My Passport/dataset/Final_dataset largest/yolov11/val/labels'
    image_folder = '/Volumes/My Passport/dataset/Final_dataset largest/images/val'
    destination_folder = '/Volumes/My Passport/dataset/Final_dataset largest/yolov11/val/images'

    # 调用函数进行操作
    copy_matching_images(txt_folder, image_folder, destination_folder)
