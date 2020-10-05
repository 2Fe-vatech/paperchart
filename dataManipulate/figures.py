import os
import sys
import cv2
import json
import argparse
from time import sleep
from tqdm import tqdm, trange
from common import cal_iou, filter_by_iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", help="base directory of images", required=True
    )
    parser.add_argument(
        "-t", "--target", help="data type [train, valid]", choices=["train", "valid"]
    )
    parser.add_argument(
        "-p",
        "--reports",
        help="results of inference",
        default="Yet-Another-EfficientDet-Pytorch/valid_bbox_results.json",
    )
    parser.add_argument("-w", "--show", help="show result image", action="store_true")
    parser.add_argument(
        "-v", "--save", help="save result image and json file", action="store_true"
    )
    parser.add_argument(
        "-l",
        "--threshold",
        help="threshold value filtering by score",
        type=float,
        default=0.5,
    )
    parser.add_argument("-r", "--results", help="results path", default="results")
    args = parser.parse_args()

    basedir = os.path.abspath(os.path.expanduser(args.basedir))
    assert os.path.isdir(basedir), basedir + " is not file or exists"

    annsfile = os.path.join(
        basedir, "annotations", "instances_{}.json".format(args.target)
    )
    assert os.path.isfile(annsfile), annsfile + " is not file or exists"

    reportfile = os.path.abspath(os.path.expanduser(args.reports))
    assert os.path.isfile(reportfile), reportfile + " is not file or exists"
    assert os.path.splitext(reportfile)[-1] == ".json", (
        reportfile + " should be json file"
    )

    with open(annsfile, "r") as jobj:
        jdata = json.load(jobj)
        assert (
            "images" in jdata and "annotations" in jdata and "categories" in jdata
        ), "[annotation file error] illegal format"

        images = jdata["images"]
        anns = jdata["annotations"]
        classes = jdata["categories"]

    imageId2Name = {}
    for image in images:
        imageId2Name[image["id"]] = image["file_name"]

    catId2Name = {}
    for cat in classes:
        catId2Name[cat["id"]] = cat["name"]

    with open(reportfile, "r") as jobj:
        reports = json.load(jobj)

    annsGroupByImageId = {}

    for report in reports:
        if report["image_id"] not in annsGroupByImageId:
            annsGroupByImageId[report["image_id"]] = []

        annsGroupByImageId[report["image_id"]].append(report)

    resultsPath = os.path.abspath(os.path.expanduser(args.results))
    os.makedirs(resultsPath, exist_ok=True)

    for annsGroupId, annsGroup in tqdm(annsGroupByImageId.items(), desc="image : "):
        filename = imageId2Name[annsGroupId]
        filepath = os.path.join(basedir, args.target, filename)
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

        filtered_anns = filter_by_iou(annsGroup)

        for ann in filtered_anns:
            if ann["score"] > args.threshold:
                image = cv2.rectangle(
                    image,
                    (int(ann["bbox"][0]), int(ann["bbox"][1])),
                    (
                        int(ann["bbox"][0] + ann["bbox"][2]),
                        int(ann["bbox"][1] + ann["bbox"][3]),
                    ),
                    (0, 0, 255),
                    1,
                )
                cv2.putText(
                    image,
                    catId2Name[ann["category_id"]],
                    (int(ann["bbox"][0]), int(ann["bbox"][1] + ann["bbox"][3] // 2)),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    0.3,
                    (255, 255, 0),
                    1,
                )

        if args.show:
            show_image = cv2.resize(image, (800, 800))
            cv2.imshow(filename, show_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if args.save:
            cv2.imwrite(os.path.join(resultsPath, filename + ".png"), image)
