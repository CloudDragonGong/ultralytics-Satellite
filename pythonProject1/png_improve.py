import cv2
import numpy as np
import os

def enhance_exposure_on_mask(src_img_path, mask_img_path, save_dir, bright_factor=1.5):
    # 1. 读取原图（RGB）和掩码图（16位单通道）
    img = cv2.imread(src_img_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_img_path, cv2.IMREAD_UNCHANGED)  # 这能读取16位灰度PNG

    if img is None or mask is None:
        print('原图或掩码图读取失败，请检查路径和格式。')
        return

    # 2. 检查mask类型是否16位
    if mask.dtype != np.uint16:
        print('警告：掩码图不是16位，当前类型为：', mask.dtype)

    # 3. 创建ID为1的掩码（严格等于1的区域）
    mask_id1 = (mask == 1).astype(np.uint8)    # H,W, 0或1

    # 4. 扩展为三通道掩码
    mask_id1_3c = np.repeat(mask_id1[:, :, np.newaxis], 3, axis=2)  # H,W,3

    # 5. 曝光增强
    img_float = img.astype(np.float32)
    img_float[mask_id1_3c == 1] *= bright_factor
    img_float = np.clip(img_float, 0, 255)
    enhanced = img_float.astype(np.uint8)

    # 6. 保存
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.basename(src_img_path)
    save_path = os.path.join(save_dir, f'enhanced_{file_name}')
    cv2.imwrite(save_path, enhanced)
    print(f'处理完成，结果已保存到：{save_path}')
# 示例调用
if __name__ == '__main__':
    src_img_path = '/Volumes/My Passport/dataset/UESD/test/images/00001.png'
    mask_img_path = '/Volumes/My Passport/dataset/UESD/test/truth_labels/00001.png'
    save_dir = '/Volumes/My Passport/dataset/UESD/test/improved_image'
    enhance_exposure_on_mask(src_img_path, mask_img_path, save_dir, bright_factor=2)