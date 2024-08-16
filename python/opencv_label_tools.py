import argparse
import numpy as np
import glob
from pathlib import Path
import cv2

# 定义命令行参数
argparser = argparse.ArgumentParser()
argparser.add_argument("--ds_dir", default="./dataset", type=str, help='raw dataset dir')
argparser.add_argument("--target_ds_dir", default="./dataset_new_del", type=str, help="new dataset dir")

args = argparser.parse_args()

# 获取旧数据集目录
old_ds_dir = Path(args.ds_dir)
imgs = old_ds_dir.glob("images/*.jpg")

# 创建目标数据集目录
target_ds_dir = Path(args.target_ds_dir)
target_img_dir = target_ds_dir / "images"
target_label_dir = target_ds_dir / "labels"
target_img_dir.mkdir(parents=True, exist_ok=True)
target_label_dir.mkdir(parents=True, exist_ok=True)

def add_label_to_img(img, label):
    H, W = img.shape[0:2]
    xywhn = label[:, 1:]
    xywh = xywhn * [[W, H, W, H]]
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    for x0, y0, x1, y1 in xyxy:
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), 2)
    return img, xyxy

def get_label_path(image_path):
    image_path = Path(image_path)
    label_path = image_path.with_suffix(".txt")
    label_path = str(label_path).replace('images/', 'labels/')
    return Path(label_path)

def get_front_obj_from_roi_save(xyxys, roi, img, ds_dir, img_stem, label_id=0):
    H, W = img.shape[:2]
    merged_img = np.zeros((640, 640, 3), dtype=np.uint8)
    cut_img_num_per_row = 640 // 32
    cut_img_max_row = 640 // 32
    for i, (x0, y0, x1, y1) in enumerate(xyxys):
        x = (x0 + x1) * 0.5
        y = (y0 + y1) * 0.5
        # cut 32x32 region around center
        x_cut0 = int(np.clip(x - 16, 0, W))
        y_cut0 = int(np.clip(y - 16, 0, H))
        x_cut1 = int(np.clip(x + 16, 0, W))
        y_cut1 = int(np.clip(y + 16, 0, H))
        img_cut = img[y_cut0:y_cut1, x_cut0:x_cut1]
        cv2.imwrite(str(ds_dir / "images" / f"{img_stem}_{i}.jpg"), img_cut)

        # label in roi
        if roi[0] < x < roi[2] and roi[1] < y < roi[3]:
            x = (x - x_cut0) / 32
            y = (y - y_cut0) / 32
            w = (x1 - x0) / 32
            h = (y1 - y0) / 32
            label = [0, x, y, w, h]
            np.savetxt(str(ds_dir / "labels" / f"{img_stem}_{i}.txt"), [label])

            # draw rectangle on img_cut
            x0 = int((x - w / 2) * 32)
            y0 = int((y - h / 2) * 32)
            x1 = int((x + w / 2) * 32)
            y1 = int((y + h / 2) * 32)
            cv2.rectangle(img_cut, (x0, y0), (x1, y1), (255, 255, 255), 2)
        cut_img_row = i // cut_img_num_per_row
        cut_img_col = i % cut_img_num_per_row
        if cut_img_row < cut_img_max_row:
            cut_img_x0 = cut_img_col * 32
            cut_img_y0 = cut_img_row * 32
            cut_img_h, cut_img_w = img_cut.shape[:2]
            merged_img[cut_img_y0:cut_img_y0 + cut_img_h,
                       cut_img_x0:cut_img_x0 + cut_img_w] = img_cut
    return merged_img

def get_front_obj_from_roi(label, roi, img):
    x0, y0, x1, y1 = roi
    H, W = img.shape[:2]
    x0, y0, x1, y1 = x0 / W, y0 / H, x1 / W, y1 / H
    new_label = []
    for i in range(len(label)):
        x, y = label[i][1:3]
        if x0 < x < x1 and y0 < y < y1:
            new_label.append(label[i])
    return new_label

def on_mouse(event, x, y, flags, param):
    global img_labeled, label, drawing, x0, y0, x1, y1

    if event == cv2.EVENT_LBUTTONDOWN:
        img_labeled = img_labeled_bac.copy()
        x0, y0 = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        x1, y1 = x, y
        cv2.rectangle(img_labeled, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imshow("labeled image", img_labeled)

cv2.namedWindow("labeled image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("labeled image", on_mouse)

cv2.namedWindow("merged image", cv2.WINDOW_NORMAL)

stop = False
for img_path in imgs:

    if stop:
        break
    label_path = get_label_path(img_path)
    label = np.loadtxt(str(label_path)).tolist()
    img = cv2.imread(str(img_path))
    img_labeled, xyxys = add_label_to_img(img.copy(), np.array(label))
    img_labeled_bac = img_labeled.copy()

    x0, y0, x1, y1 = 0, 0, 0, 0
    while True:
        cv2.imshow("labeled image", img_labeled)
        key = cv2.waitKey(10) & 0xFF

        if key == 27:  # 按下ESC键退出
            stop = True
            break

        elif key == 32:  # 按下空格键继续到下一张图片
            break

        elif key == ord('s'):  # 按下S键保存并退出
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
            merged_img = get_front_obj_from_roi_save(xyxys, (x0, y0, x1, y1), img, target_ds_dir, img_path.stem)
            cv2.imshow("merged image", merged_img)
            break

cv2.destroyAllWindows()
