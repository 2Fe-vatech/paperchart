import os
import sys
import csv
import cv2
import json
import numpy as np
import argparse
from tqdm import tqdm
from common import cal_iou, filter_by_iou


def evalutation(Anns1, Anns2):
    results = []
    ious = []
    found = 0
    notFound = 0

    for ann1 in Anns1:
        index = -1
        max_iou = 0.1

        for idx, ann2 in enumerate(Anns2):
            iou = cal_iou(ann1["bbox"], ann2["bbox"])

            if max_iou < iou:
                index = idx
                max_iou = iou

        if index != -1:
            results.append(Anns2[index])
            ious.append(max_iou)
            found += 1
        else:
            notFound += 1
            results.append({"bbox": [np.nan, np.nan, np.nan, np.nan]})
            ious.append(np.nan)

    return results, ious, found, notFound


def sortAnns(Anns):
    sorted_index = np.array([ann["bbox"][1] for ann in Anns]).argsort()
    sorted_anns = np.array(Anns)[sorted_index]
    return sorted_anns


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

    imageId2Anns = {}
    for ann in anns:
        if ann["image_id"] not in imageId2Anns.keys():
            imageId2Anns[ann["image_id"]] = []

        imageId2Anns[ann["image_id"]].append(ann)

    imageId2Name = {}
    for image in images:
        imageId2Name[image["id"]] = image["file_name"]

    with open(reportfile, "r") as jobj:
        reports = json.load(jobj)

    annsGroupByImageId = {}

    for report in reports:
        if report["image_id"] not in annsGroupByImageId:
            annsGroupByImageId[report["image_id"]] = []

        annsGroupByImageId[report["image_id"]].append(report)

    resultsPath = os.path.abspath(os.path.expanduser(args.results))
    os.makedirs(resultsPath, exist_ok=True)

    raw_results = {}

    totalFound = 0
    totalNotFound = 0
    totalFoundWrong = 0
    totalious = []
    rawious = {}

    for annsGroupId, annsGroup in tqdm(annsGroupByImageId.items(), desc="image : "):
        filtered_anns = filter_by_iou(annsGroup)
        filtered_anns = [ann for ann in filtered_anns if ann["score"] >= 0.1]

        orderedResults, ious, found0, notFound = evalutation(
            imageId2Anns[annsGroupId], filtered_anns
        )
        _, _, found1, foundWrong = evalutation(filtered_anns, imageId2Anns[annsGroupId])

        assert found0 == found1, "found0 and found1 should be same"

        totalFound += found1
        totalNotFound += notFound
        totalFoundWrong += foundWrong
        totalious.extend(ious)
        rawious[annsGroupId] = ious
        raw_results[annsGroupId] = orderedResults

    totalious = np.array(totalious)
    iouMean = np.nanmean(totalious)
    iouStd = np.nanstd(totalious)
    iouMax = np.nanmax(totalious)
    iouMin = np.nanmin(totalious)

    with open(f"{resultsPath}/result_raw.csv", "w") as cobj:
        csvwriter = csv.writer(cobj, delimiter=",")
        csvwriter.writerow(["filename", "idx", "origin_bbox", "pred_bbox", "iou"])

        for imageid in imageId2Name.keys():
            for idx, result in enumerate(raw_results[imageid]):
                for jdx, bbox in enumerate(result["bbox"]):
                    result["bbox"][jdx] = round(bbox, 2)

                csvwriter.writerow(
                    [imageId2Name[imageid], idx]
                    + [imageId2Anns[imageid][idx]["bbox"]]
                    + [result["bbox"]]
                    + [rawious[imageid][idx]]
                )

    with open(f"{resultsPath}/result_summary.csv", "w") as cobj:
        csvwriter = csv.writer(cobj, delimiter=",")
        csvwriter.writerow(["found summary"])
        csvwriter.writerow(
            [
                "total found",
                totalFound,
                "total not found",
                totalNotFound,
                "total wrong found",
                totalFoundWrong,
                "total instance",
                totalFound + totalNotFound,
            ]
        )
        csvwriter.writerow(["iou summary"])
        csvwriter.writerow(
            ["mean", iouMean, "std", iouStd, "max", iouMax, "min", iouMin]
        )
