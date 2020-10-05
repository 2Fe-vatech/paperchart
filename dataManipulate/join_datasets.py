import os
import json
import argparse

from tqdm import tqdm
from shutil import copy2
from common import loadAnns, saveAnns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d0", "--dataset0", help="dataset0 directory to join", required=True
    )
    parser.add_argument(
        "-d1", "--dataset1", help="dataset1 directory to join", required=True
    )
    parser.add_argument(
        "-n", "--name", help="append name for replace image", default="d3"
    )
    parser.add_argument(
        "-t", "--target", help="data type [train, valid]", choices=["train", "valid"]
    )
    parser.add_argument(
        "-o", "--output", help="save result file path", default="results"
    )
    args = parser.parse_args()

    images0, anns0, classes0 = loadAnns(args.dataset0, args.target)
    images1, anns1, classes1 = loadAnns(args.dataset1, args.target)

    assert classes0 == classes1, "two dataset has different classes"

    imagename2imgidx = {}
    for idx, image1 in enumerate(images1):
        imagename2imgidx[image1["file_name"]] = idx

    imageid2annidx = {}
    for idx, ann1 in enumerate(anns1):
        if ann1["image_id"] not in imageid2annidx:
            imageid2annidx[ann1["image_id"]] = []
        imageid2annidx[ann1["image_id"]].append(idx)

    os.makedirs(os.path.join(args.output, args.target), exist_ok=True)
    listdb0 = [
        fn
        for fn in os.listdir(os.path.join(args.dataset0, args.target))
        if os.path.splitext(fn)[-1] in [".bmp", ".BMP", ".jpg"]
    ]
    listdb1 = [
        fn
        for fn in os.listdir(os.path.join(args.dataset1, args.target))
        if os.path.splitext(fn)[-1] in [".bmp", ".BMP", ".jpg"]
    ]

    for filename in listdb0:
        copy2(
            os.path.join(args.dataset0, args.target, filename),
            os.path.join(args.output, args.target),
        )

    for filename in listdb1:
        if filename in listdb0:
            basename, ext = os.path.splitext(filename)
            replacename = basename + "." + args.name + ext
            images1[imagename2imgidx[filename]]["file_name"] = replacename
            copy2(
                os.path.join(args.dataset1, args.target, filename),
                os.path.join(args.output, args.target, replacename),
            )
        else:
            copy2(
                os.path.join(args.dataset1, args.target, filename),
                os.path.join(args.output, args.target),
            )

    lastimageid = images0[-1]["id"]
    for image1 in images1:
        lastimageid += 1

        for ann1Idx in imageid2annidx[image1["id"]]:
            anns1[ann1Idx]["image_id"] = lastimageid

        image1["id"] = lastimageid
        images0.append(image1)

    lastannid = anns0[-1]["id"]
    for ann1 in anns1:
        lastannid += 1

        ann1["id"] = lastannid
        anns0.append(ann1)

    os.makedirs(os.path.join(args.output, "annotations"), exist_ok=True)
    saveAnns(
        os.path.join(
            args.output, "annotations", "instances_{}.json".format(args.target)
        ),
        {"images": images0, "annotations": anns0, "categories": classes0},
    )
