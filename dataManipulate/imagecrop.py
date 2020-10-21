import os
import cv2
import sys
import json
import argparse
import numpy as np
from common import getListCenter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", help="base directory of images", required=True
    )
    parser.add_argument(
        "-t", "--target", help="data type [train, valid]", choices=["train", "valid"]
    )
    parser.add_argument(
        "-p", "--padding", help="padding on crop image", type=int, default=0
    )
    parser.add_argument("-w", "--show", help="show result image", action="store_true")
    parser.add_argument(
        "-v", "--save", help="save result image and json file", action="store_true"
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

    id2catid = {}
    for ann in anns:
        id2catid[ann["id"]] = ann["category_id"]

    dateboxlist = {}
    bboxidxlist = {}
    bboxlist = {}

    imageids = 10000
    annotationids = 30000

    images = []
    annotations = []

    categories = cats
    assert len(categories) == 11, "categories length should be 11"
    del categories[10]

    os.makedirs(os.path.join(args.results, "annotations"), exist_ok=True)

    old = sys.stdout
    sys.stdout = open(
        os.path.join(
            args.results, "annotations", "instances_{}.json".format(args.target)
        ),
        "w",
    )

    for ann in anns:
        if ann["category_id"] == 11:
            if ann["image_id"] not in dateboxlist.keys():
                dateboxlist[ann["image_id"]] = []
            dateboxlist[ann["image_id"]].append({"bbox": ann["bbox"]})
        else:
            if ann["image_id"] not in bboxlist.keys():
                bboxlist[ann["image_id"]] = []
                bboxidxlist[ann["image_id"]] = []
            bboxidxlist[ann["image_id"]].append(ann["id"])
            bboxlist[ann["image_id"]].append(ann["bbox"])

    for imageid, datebox in dateboxlist.items():
        bboxcenter = getListCenter(bboxlist[imageid])
        for idx, db in enumerate(datebox):
            x_low = np.where(bboxcenter[:, 0] > db["bbox"][0])[0]
            x_high = np.where(bboxcenter[:, 0] < (db["bbox"][0] + db["bbox"][2]))[0]
            y_low = np.where(bboxcenter[:, 1] > db["bbox"][1])[0]
            y_high = np.where(bboxcenter[:, 1] < (db["bbox"][1] + db["bbox"][3]))[0]

            subbox = []
            for i in range(len(bboxcenter)):
                if i in x_low and i in x_high and i in y_low and i in y_high:
                    subbox.append(
                        {id2catid[bboxidxlist[imageid][i]]: bboxlist[imageid][i]}
                    )

            dateboxlist[imageid][idx]["subbox"] = subbox

    for imageid, dateboxes in dateboxlist.items():
        filepath = os.path.join(basedir, args.target, imageId2Name[imageid])
        image = cv2.imread(filepath)
        for num, datebox in enumerate(dateboxes):
            dbbox = np.asarray(datebox["bbox"], dtype=np.int)
            croppedimage = image[
                dbbox[1] - args.padding : dbbox[1] + dbbox[3] + args.padding,
                dbbox[0] - args.padding : dbbox[0] + dbbox[2] + args.padding,
            ]
            filenamebase, fileext = os.path.splitext(imageId2Name[imageid])
            filename = "{}_{}{}".format(filenamebase, num, fileext)

            if args.save:
                cv2.imwrite(os.path.join(args.results, filename), croppedimage)

            imageids = imageids + 1
            images.append(
                {
                    "file_name": filename,
                    "height": croppedimage.shape[0],
                    "width": croppedimage.shape[1],
                    "id": imageids,
                }
            )

            subboxes = datebox["subbox"]
            for subbox in subboxes:
                annotationids = annotationids + 1
                sbbox = list(subbox.values())[0]
                sbbox[0] = (
                    sbbox[0] - dbbox[0] + args.padding
                    if sbbox[0] - dbbox[0] >= 0
                    else 0.0
                )
                sbbox[1] = (
                    sbbox[1] - dbbox[1] + args.padding
                    if sbbox[1] - dbbox[1] >= 0
                    else 0.0
                )

                annotations.append(
                    {
                        "iscrowd": 0,
                        "image_id": imageids,
                        "bbox": sbbox,
                        "area": sbbox[2] * sbbox[3],
                        "category_id": list(subbox.keys())[0],
                        "id": annotationids,
                    }
                )

    print(
        json.dumps(
            {"images": images, "annotations": annotations, "categories": categories},
            indent=2,
        )
    )

    sys.stdout = old
