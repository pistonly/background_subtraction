import argparse
import numpy as np
from pathlib import Path
import cv2

# 定义命令行参数
argparser = argparse.ArgumentParser()
argparser.add_argument("--roi_ds_dir", type=str, default="./dataset_new", help="source dataset")
argparser.add_argument("--merged_ds_dir", type=str, default="./merge_dataset", help="target dataset")
args = argparser.parse_args()

# 定义数据集路径
roi_ds_dir = Path(args.roi_ds_dir)
target_ds_dir = Path(args.merged_ds_dir)
target_img_dir = target_ds_dir / "images"
target_label_dir = target_ds_dir / "labels"
target_img_dir.mkdir(parents=True, exist_ok=True)
target_label_dir.mkdir(parents=True, exist_ok=True)

# 获取图像路径
imgs = roi_ds_dir.glob("images/*.jpg")

img_grouped = {}

# 将图像按组分类
for img in imgs:
    stem = img.stem
    stem_info = stem.split("_")
    if stem_info[1] not in img_grouped:
        img_grouped[stem_info[1]] = []
    img_grouped[stem_info[1]].append(img)

# 合并每组图像为 640x640 大小的图像
roi_per_row = 640 // 32
roi_row_max = 640 // 32

for g, img_list in img_grouped.items():
    target_img = np.zeros((640, 640, 3), dtype=np.uint8) + 114
    labels = []
    for i, img_path in enumerate(img_list):
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]
        roi_row = i // roi_per_row
        roi_col = i % roi_per_row
        roi_x0 = roi_col * 32
        roi_y0 = roi_row * 32
        if roi_row < roi_row_max:
            target_img[roi_y0:roi_y0 + H, roi_x0:roi_x0 + W] = img
        else:
            print(f"ROI exceeds the maximum of {roi_per_row * roi_row_max}")
            break

        img_label_path = img_path.with_suffix(".txt")
        img_label_path = str(img_label_path).replace("images/", "labels/")
        if Path(img_label_path).is_file():
            label = np.loadtxt(img_label_path)
            if label.ndim == 1:
                label = label[np.newaxis, :]
            xywh = label[:, 1:] * [[W, H, W, H]]
            xywh_new = xywh / 640
            label_new = np.hstack([label[:, 0:1], xywh_new])
            label_new[:, 1] = (label_new[:, 1] + roi_x0 / 640)
            label_new[:, 2] = (label_new[:, 2] + roi_y0 / 640)
            labels.append(label_new)

    if labels:
        labels = np.vstack(labels)
        np.savetxt(str(target_label_dir / f"{g}.txt"), labels, fmt="%.6f")

    cv2.imwrite(str(target_img_dir / f"{g}.jpg"), target_img)

