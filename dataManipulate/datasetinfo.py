import os
import argparse
import cv2
from common import getimageinfo

from tqdm import tqdm


def getImageSizesInDataset(basedir, targets):
    imgSizes = []

    for target in targets:
        assert os.path.isdir(os.path.join(basedir, target)), (
            os.path.join(basedir, target) + " is not a directory"
        )

        for filename in tqdm(
            os.listdir(os.path.join(basedir, target)), desc=f"{target} : "
        ):
            if os.path.splitext(filename)[-1] in [".jpg", ".png", ".bmp", ".BMP"]:
                _, w, h, _ = getimageinfo(os.path.join(basedir, target, filename))

                if (w, h) not in imgSizes:
                    imgSizes.append((w, h))

    return imgSizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", help="base directory of dataset", required=True
    )
    parser.add_argument(
        "-t",
        "--targets",
        help="subcategory of datasets",
        nargs="+",
        default=["train", "valid"],
    )
    args = parser.parse_args()

    basedir = os.path.abspath(os.path.expanduser(args.basedir))

    imgSizes = getImageSizesInDataset(basedir, args.targets)

    for idx, size in enumerate(imgSizes):
        print(f"{idx} : {size}")
