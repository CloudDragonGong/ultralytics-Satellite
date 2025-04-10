import os


def count_images_in_folder(folder_path):
    # 定义常见的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    image_count = 0

    # 遍历文件夹中的文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件扩展名是否在图片扩展名列表中
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_count += 1

    return image_count


if __name__ == "__main__":
    # 输入文件夹地址
    folder_path = "/Volumes/My Passport/dataset/UESD_edition2"

    if os.path.isdir(folder_path):
        count = count_images_in_folder(folder_path)
        print(f"文件夹中图片的数量为: {count}")
    else:
        print("请输入有效的文件夹地址。")