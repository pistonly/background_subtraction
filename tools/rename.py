from pathlib import Path

img_dir = Path("/home/liuyang/datasets/sod4bird/35/")
for img_f in img_dir.iterdir():
    img_stem = img_f.stem
    img_name_new = f"{int(img_f.stem):04d}.jpg"
    img_f.rename(str(img_dir / img_name_new))
