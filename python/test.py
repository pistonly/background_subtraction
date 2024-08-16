import cv2
import numpy as np
import time


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


    frame_diff = FrameDifference(threshold=15)
    bin_file = "/home/liuyang/Documents/haisi/my_samples/tools/out_1920x1080.bin"
    with open(bin_file, "rb") as f:
        imgs = f.read()
        size = len(imgs)
        img_num = size // (1080 * 1920)
        imgs = np.ndarray((img_num, 1080, 1920), dtype=np.uint8, buffer=imgs)

    for img in imgs:
        img_foreground = frame_diff.process(img)
        if img_foreground is not None:
            cv2.imshow("img_foreground", img_foreground)
            cv2.imshow("img_input", img)
            time.sleep(0.5)

            if 0xFF & cv2.waitKey(10) == 27:
                break

    cv2.destroyAllWindows()

