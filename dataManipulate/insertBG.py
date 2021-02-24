import os
import cv2
import random
import argparse
import numpy as np
from tqdm import tqdm
from common import display_image, getimageinfo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", help="base directory of images", required=True
    )
    parser.add_argument(
        "-t", "--target", help="data type [train, valid]", choices=["train", "valid"]
    )
    parser.add_argument(
        "-g", "--bgdir", help=" directory of background images", default="datasets"
    )
    parser.add_argument("-w", "--show", help="show result image", action="store_true")
    parser.add_argument(
        "-v", "--save", help="save result image and json file", action="store_true"
    )
    args = parser.parse_args()

    basedir = os.path.abspath(os.path.expanduser(args.basedir))
    assert os.path.isdir(basedir), basedir + " is not a directory"

    targetdir = os.path.abspath(
        os.path.expanduser(os.path.join(args.basedir, args.target))
    )
    assert os.path.isdir(targetdir), targetdir + " is not a directory"

    bgdir = os.path.abspath(os.path.expanduser(args.bgdir))
    assert os.path.isdir(bgdir), bgdir + " is not a directory"

    imglist = [
        filename
        for filename in os.listdir(targetdir)
        if os.path.splitext(filename)[-1] in [".jpg", ".bmp", ".BMP"]
    ]
    bglist = [
        os.path.join(bgdir, filename)
        for filename in os.listdir(bgdir)
        if os.path.splitext(filename)[-1] in [".jpg", ".bmp", ".BMP"]
    ]

    result_path = os.path.join(basedir, args.target + "_noisy")
    os.makedirs(result_path, exist_ok=True)

    for imagename in tqdm(imglist):
        bgimagepath = bglist[random.randint(0, len(bglist) - 1)]
        bgimage, bgimage_w, bgimage_h, bgimage_area = getimageinfo(bgimagepath)

        imagepath = os.path.join(targetdir, imagename)
        baseimage, baseimage_w, baseimage_h, baseimage_area = getimageinfo(imagepath)

        if baseimage_area > bgimage_area:
            bgimage = cv2.resize(bgimage, (baseimage_w, baseimage_h))
        else:
            baseimage = cv2.resize(baseimage, (bgimage_w, bgimage_h))

        assert baseimage.shape == bgimage.shape, (
            str(baseimage.shape) + " != " + str(bgimage.shape)
        )

        bgimage = cv2.bitwise_not(bgimage)
        baseimage = cv2.bitwise_not(baseimage)
        resultimage = np.asarray(bgimage, dtype=int) + np.asarray(baseimage, dtype=int)
        resultimage = np.asarray(np.clip(resultimage, 0, 255), dtype=np.uint8)
        resultimage = cv2.bitwise_not(resultimage)

        if args.save:
            cv2.imwrite(os.path.join(result_path, imagename), resultimage)

        if args.show:
            display_image(imagename, resultimage, size=(800, 600))
