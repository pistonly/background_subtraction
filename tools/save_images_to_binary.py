import cv2
import numpy as np
from pathlib import Path

def save_images_to_binary(file_paths, output_file):
    # 确保所有输入路径都被包含在列表中
    
    with open(output_file, 'wb') as f:
        for path in file_paths:
            # 使用OpenCV读取图片并转换为灰度
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # 确保图片尺寸正确
            if img.shape != (720, 1280):
                raise ValueError(f"图片 {path} 的尺寸不是1920x1080")
            # 将图片数据写入二进制文件
            img.tofile(f)

img_dir = Path("/home/liuyang/datasets/sod4bird/35_6600-6800_1280x720roi")
img_names = [f.name for f in img_dir.iterdir()]
img_names.sort()
image_paths = [str(img_dir / img_n) for img_n in img_names]

# 输出文件路径
output_path = 'output_images_6600-6800_1280x720.bin'

# 执行函数
save_images_to_binary(image_paths, output_path)
