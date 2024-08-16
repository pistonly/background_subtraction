import cv2
from pathlib import Path

img_dir = Path("/home/liuyang/datasets/sod4bird/35/")

# roi_ul = (1273, 180)
# roi_sz = (1920, 1080)
roi_ul = [1273, 972]
roi_sz = [1280, 720]

output_dir = Path("/home/liuyang/datasets/sod4bird/35_6600-6800_1280x720roi_moving")
output_dir.mkdir(exist_ok=True, parents=True)
imgs = [f.name for f in img_dir.iterdir()]
imgs.sort()

for i, img_ in enumerate(imgs):
    if i >= 6600 and i < 6800:
        img = cv2.imread(str(img_dir / img_))
        if img is not None:
            img_roi = img[roi_ul[1]:roi_ul[1] + roi_sz[1],
                          roi_ul[0]:roi_ul[0] + roi_sz[0]]
            cv2.imwrite(str(output_dir / img_), img_roi)

            # moving roi
            roi_ul[0] -= 1
        else:
            print(f"img: {str(img_)}: read error")
