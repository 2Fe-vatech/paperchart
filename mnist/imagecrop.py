import os
import cv2
import json
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", help="base directory of images", required=True
    )
    parser.add_argument(
        "-t", "--target", help="data type [train, valid]", choices=["train", "valid"]
    )
    parser.add_argument("-r", "--results", help="results path", default="croppedimages")
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

    imageId2Name = {}
    for image in images:
        imageId2Name[image["id"]] = image["file_name"]

    catId2Name = {}
    for cat in cats:
        catId2Name[cat["id"]] = cat["name"]

    imageId2ann = {}
    for ann in anns:
        if ann["image_id"] not in imageId2ann.keys():
            imageId2ann[ann["image_id"]] = []
        imageId2ann[ann["image_id"]].append(ann)

    resultsdir = os.path.join("datasets", args.results)
    os.makedirs(resultsdir, exist_ok=True)

    for imgid, annsbyimg in imageId2ann.items():
        assert len(annsbyimg) == 8, "not enough labels : " + str(imgid)

        img = cv2.imread(os.path.join(basedir, args.target, imageId2Name[imgid]))
        for ann in annsbyimg:
            bbox = np.array(ann["bbox"], dtype=int)
            croppedimage = img[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]

            cv2.imwrite(
                os.path.join(
                    resultsdir,
                    "{}3{}_{}.png".format(
                        args.target, ann["id"], ann["category_id"] - 1
                    ),
                ),
                croppedimage,
            )
