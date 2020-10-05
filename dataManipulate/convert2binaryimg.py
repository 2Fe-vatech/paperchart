import os
import cv2
import argparse
import numpy as np
from shutil import copy2
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", help="base directory of images", required=True
    )
    parser.add_argument(
        "-o", "--output", help="save result file path", default="results"
    )
    args = parser.parse_args()

    basedir = os.path.abspath(os.path.expanduser(args.basedir))
    assert os.path.isdir(basedir), basedir + " is not directory"

    os.makedirs(os.path.join(args.output, "paperchart"), exist_ok=True)

    kernel = np.ones((3, 3), np.uint8)

    for dirpath, dirname, filenames in os.walk(basedir):
        outdir = os.path.join(args.output, "paperchart" + dirpath.replace(basedir, ""))
        os.makedirs(outdir, exist_ok=True)
        for filename in filenames:
            if os.path.splitext(filename)[-1] in [".jpg", ".png", ".bmp", ".BMP"]:
                image = cv2.imread(os.path.join(dirpath, filename))
                gray_images = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                img_binary = cv2.adaptiveThreshold(
                    gray_images,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    21,
                    5,
                )

                # img_binary = cv2.erode(img_binary, kernel, iterations=1)
                # img_binary = cv2.dilate(img_binary, kernel, iterations=2)

                img_binary = cv2.cvtColor(img_binary, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(os.path.join(outdir, filename), img_binary)
            else:
                copy2(os.path.join(dirpath, filename), outdir)
