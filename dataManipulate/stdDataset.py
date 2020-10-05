import os
import cv2
import argparse
import numpy as np

from tqdm import tqdm
from common import loadAnns, saveAnns
from datasetinfo import getImageSizesInDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", help="base directory of images", required=True
    )
    parser.add_argument(
        "-t",
        "--targets",
        help="subcategory of datasets",
        nargs="+",
        default=["train", "valid"],
    )
    parser.add_argument(
        "-o", "--output", help="save result file path", default="results"
    )
    args = parser.parse_args()

    basedir = os.path.abspath(os.path.expanduser(args.basedir))
    # imageSizes = getImageSizesInDataset(basedir, args.targets)

    # small_area = 0

    # for idx, size in enumerate(imageSizes):
    #     area = size[0] * size[1]

    #     if idx == 0 or area < small_area:
    #         small_area = area
    #         std_size = imageSizes[idx]

    imageSizes = [(2480, 3507), (793, 1121)]
    std_size = (793, 1121)

    convertRate = {}
    for size in imageSizes:
        assert size[0] != 0 and size[1] != 0, "size should not be zero"
        convertRate[size[0]] = std_size[0] / size[0]

    for target in args.targets:
        imageAnns, Anns, categories = loadAnns(basedir, target)

        idlist = {}
        os.makedirs(os.path.join(args.output, target), exist_ok=True)

        for imageann in tqdm(imageAnns, desc=f"{target} : "):
            idlist[imageann["id"]] = convertRate[imageann["width"]]
            imageann["height"] = int(
                imageann["height"] * convertRate[imageann["width"]] + 0.5
            )
            imageann["width"] = int(
                imageann["width"] * convertRate[imageann["width"]] + 0.5
            )
            img = cv2.imread(os.path.join(basedir, target, imageann["file_name"]))
            img = cv2.resize(img, (imageann["width"], imageann["height"]))

            cv2.imwrite(os.path.join(args.output, target, imageann["file_name"]), img)

        for ann in Anns:
            ann["bbox"] = np.array(ann["bbox"]) * idlist[ann["image_id"]]
            ann["bbox"] = np.array(ann["bbox"]).tolist()

        os.makedirs(os.path.join(args.output, "annotations"), exist_ok=True)
        saveAnns(
            os.path.join(args.output, "annotations", f"instances_{target}.json"),
            {"images": imageAnns, "annotations": Anns, "categories": categories},
        )
