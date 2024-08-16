import cv2
import numpy as np


class FrameDifference(object):
    def __init__(self, enableThreshold=True, threshold=15) -> None:
        self.enableThreshold = enableThreshold
        self.threshold = threshold
        self.img_background = None
        self.img_foreground = np.empty((10, 10), dtype=np.uint8)

    def process(self, img_input: np.ndarray):
        if (self.img_background is None):
            self.img_background = img_input.copy()
            return

        self.img_foreground = cv2.absdiff(self.img_background, img_input)

        if self.enableThreshold:
            ret, self.img_foreground = cv2.threshold(self.img_foreground, self.threshold, 255, cv2.THRESH_BINARY)

        self.img_background = img_input.copy()


        return self.img_foreground



if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import time


    argparser = argparse.ArgumentParser()
    argparser.add_argument("--image_dir", default="", type=str, help="input dir for images")
    argparser.add_argument("--threshold", default=15, type=int, help="threshold for foreground")
    argparser.add_argument("--is_video", action='store_true', help="get frames from video")
    args = argparser.parse_args()

    frame_diff = FrameDifference(threshold=args.threshold)
    cost_times = []

    if args.is_video:
        capture = cv2.VideoCapture(args.image_dir)
        if not capture.isOpened():
            print("Error: Cannot open video.")
        else:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Total number of frames: {total_frames}")

        img_num = 0
        for _ in range(total_frames):
            flag, frame = capture.read()
            if flag:
                t0 = time.time()
                img_foreground = frame_diff.process(frame)
                if img_foreground is not None:
                    cv2.imshow("img_foreground", img_foreground)
                    cv2.imshow("img_input", frame)
                    # cv2.imwrite(f"img_{img_num:04d}.jpg", img_foreground)
                    img_num += 1

                    if 0xFF & cv2.waitKey(10) == 27:
                        break
    else:
        img_array = sorted([str(img_i) for img_i in Path(args.image_dir).iterdir()])
        img_num = 0
        for img_i in img_array:
            img = cv2.imread(img_i)
            t0 = time.time()
            img_foreground = frame_diff.process(img)
            if img_foreground is not None:
                cv2.imshow("img_foreground", img_foreground)
                cv2.imshow("img_input", img)
                cv2.imwrite(f"img_{img_num:04d}.jpg", img_foreground)
                img_num += 1

                if 0xFF & cv2.waitKey(10) == 27:
                    break

    cv2.destroyAllWindows()
