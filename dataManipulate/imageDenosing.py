import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

def erase_lines(img, direction):
    assert direction in ["horizontal", "vertical"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    clean = thresh.copy()

    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (1,30) if direction == "vertical" else (15,1))
    detect = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernal, iterations=2)
    cnts = cv2.findContours(detect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        cv2.drawContours(clean, [c], -1, 0, 2)

    return clean

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
            img = cv2.imread(os.path.join(targetdir, filename), cv2.IMREAD_COLOR)
            # dst = cv2.bitwise_not(dst)
            # dst = cv2.dilate(img, kernel, iterations=1)
            # dst = cv2.erode(dst, kernel, iterations=1)
            # img =cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            # dst = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
            # lines = houghLines(img, 140)
            # dst = cv2.Canny(img, 50, 150, apertureSize=3)
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = erase_lines(img, direction="vertical")

            if args.save:
                cv2.imwrite(os.path.join(args.results, filename), img)
