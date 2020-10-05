import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm

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
    assert os.path.isdir(basedir), basedir + " is not file or exists"

    annsfile = os.path.join(
        basedir, "annotations", "instances_{}.json".format(args.target)
    )
    assert os.path.isfile(annsfile), annsfile + " is not file or exists"

    with open(annsfile, "r") as jobj:
        jdata = json.load(jobj)
        assert (
            "images" in jdata and "annotations" in jdata and "categories" in jdata
        ), "[annotation file error] illegal format"

        images = jdata["images"]
        anns = jdata["annotations"]
        cats = jdata["categories"]

    catid2name = {}

    for cat in cats:
        catid2name[cat["id"]] = cat["name"]

    bboxperimage = {}
    for ann in anns:
        if ann["image_id"] not in bboxperimage.keys():
            bboxperimage[ann["image_id"]] = []

        bboxperimage[ann["image_id"]].append(
            {catid2name[ann["category_id"]]: ann["bbox"]}
        )

    os.makedirs(args.results, exist_ok=True)

    for image in tqdm(images):
        # if image['file_name'] == "2023.03.17.jpg":
        img = cv2.imread(os.path.join(basedir, args.target, image["file_name"]))

        bboxes = bboxperimage[image["id"]]

        for idx, bbox in enumerate(bboxes):
            for key, bb in bbox.items():
                pass
            cv2.rectangle(
                img,
                (int(bb[0]), int(bb[1])),
                (int(bb[0] + bb[2]), int(bb[1] + bb[3])),
                (0, 0, 255),
                1,
            )
            if key != "date":
                cv2.putText(
                    img,
                    key,
                    (int(bb[0]), int(bb[1]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.2,
                    (0, 0, 0),
                    1,
                )
            cv2.imwrite(os.path.join(args.results, image["file_name"]), img)
            # label = list(bbox.keys())[0]
            # bb = list(bbox.values())[0]
            # cropimage = img[int(bb[1]):int(bb[1]+bb[3]), int(bb[0]): int(bb[0]+bb[2]), :]
            # cv2.imwrite(os.path.join(args.results, "{}_{}{}.png".format(image['file_name'], idx, label)), cropimage)
