import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm


def houghLines(img, thr):
    dst = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, thr)

    new_lines = []
    for line in lines:
        r, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y1 + 1000 * a)

        new_lines.append([(x1, y1), (x2, y2)])

    return new_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", help="base directory of images", required=True
    )
    parser.add_argument(
        "-t", "--target", help="data type [train, valid]", choices=["train", "valid"]
    )
    parser.add_argument("-w", "--show", help="show result image", action="store_true")
    parser.add_argument(
        "-v", "--save", help="save result image and json file", action="store_true"
    )
    parser.add_argument("-r", "--results", help="results path", default="results")
    args = parser.parse_args()

    basedir = os.path.abspath(os.path.expanduser(args.basedir))
    assert os.path.isdir(basedir), basedir + " is not a directory"

    targetdir = os.path.abspath(
        os.path.expanduser(os.path.join(args.basedir, args.target))
    )
    assert os.path.isdir(targetdir), targetdir + " is not a directory"

    os.makedirs(args.results, exist_ok=True)

    kernel = np.ones((3, 3), np.uint8)
    for filename in tqdm(os.listdir(targetdir)):
        if os.path.splitext(filename)[-1] in [".bmp", ".jpg", ".BMP"]:
            img = cv2.imread(os.path.join(targetdir, filename), cv2.IMREAD_GRAYSCALE)
            # dst = cv2.bitwise_not(dst)
            # dst = cv2.dilate(img, kernel, iterations=1)
            # dst = cv2.erode(dst, kernel, iterations=1)
            # img =cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            # dst = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
            # lines = houghLines(img, 140)
            dst = cv2.Canny(img, 50, 150, apertureSize=3)
            if args.save:
                cv2.imwrite(os.path.join(args.results, filename), dst)
