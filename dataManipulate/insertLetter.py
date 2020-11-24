import os
import cv2
import random
import argparse
import numpy as np
from tqdm import tqdm
from common import display_image, getimageinfo

def addImage():
    pass


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
    parser.add_argument(
        "-r", "--resultdir", help=" directory of results images", default="results"
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
        if os.path.splitext(filename)[-1] in [".jpg", ".bmp", ".BMP", ".png"]
    ]
    bglist = [
        os.path.join(bgdir, filename)
        for filename in os.listdir(bgdir)
        if os.path.splitext(filename)[-1] in [".jpg", ".bmp", ".BMP", ".png"]
    ]

    result_path = os.path.join(args.resultdir)
    os.makedirs(result_path, exist_ok=True)

    for imagename in tqdm(imglist):
        imagepath = os.path.join(targetdir, imagename)
        baseimage, baseimage_w, baseimage_h, baseimage_area = getimageinfo(imagepath)
        
        for bgimagepath in bglist:
        # bgimagepath = bglist[random.randint(0, len(bglist) - 1)]
            bgimage, bgimage_w, bgimage_h, bgimage_area = getimageinfo(bgimagepath)

            bgscale = bgimage_w/bgimage_h
            
            new_bgimage = np.ones((baseimage_h, baseimage_w, 3), dtype=np.uint8) * 255
            
            loc_w = random.randint(0, baseimage_w - int(300*bgscale))
            loc_h = random.randint(0, baseimage_h - 300)
            
            new_bgimage[loc_h:loc_h+300,loc_w:loc_w + int(300*bgscale),:] = cv2.resize(bgimage, (int(300*bgscale), 300))
            
            new_bgimage = cv2.bitwise_not(np.array(new_bgimage))
            
            baseimage = cv2.bitwise_not(baseimage)
            baseimage = np.asarray(new_bgimage, dtype=int) + np.asarray(baseimage, dtype=int)
            baseimage = np.asarray(np.clip(baseimage, 0, 255), dtype=np.uint8)
            baseimage = cv2.bitwise_not(baseimage)
        cv2.imwrite("tests.png", baseimage)
        # if args.save:
        #     cv2.imwrite(os.path.join(result_path, imagename), resultimage)

        # if args.show:
        #     display_image(imagename, resultimage, size=(800, 600))
