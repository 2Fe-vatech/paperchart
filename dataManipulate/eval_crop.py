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
    correct = np.zeros(10, dtype=int)
    incorrect = np.zeros(10, dtype=int)
    ious = np.zeros(10, dtype=float)
    not_exist = np.zeros(10, dtype=int)

    for loc, ann1 in enumerate(Anns1):
        index = -1
        max_iou = 0.5

        for idx, ann2 in enumerate(Anns2):
            iou = cal_iou(ann1["bbox"], ann2["bbox"])
            if max_iou < iou:
                index = idx
                max_iou = iou

        if index != -1:
            ious[loc] = max_iou

            if Anns2[index]["category_id"] == ann1["category_id"]:
                correct[loc] += 1
            else:
                incorrect[loc] += 1
        else:
            ious[loc] = np.nan
            not_exist[loc] += 1

    return correct, incorrect, ious, not_exist


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

    total_TP = np.zeros(8, dtype=int)
    total_FP = np.zeros(8, dtype=int)
    total_TN = np.zeros(8, dtype=int)
    total_FN = np.zeros(10, dtype=int)
    total_ious = []
    raw_results = {}
    sentence_correct = 0
    sentence_incorrect = 0

    for annsGroupId, annsGroup in tqdm(annsGroupByImageId.items(), desc="image : "):
        filtered_anns = filter_by_iou(annsGroup)
        filtered_anns = [ann for ann in filtered_anns if ann["score"] >= 0.3]

        sorted_index = np.array(
            [filtered_ann["bbox"][0] for filtered_ann in filtered_anns]
        ).argsort()
        filtered_anns = np.array(filtered_anns)[sorted_index]

        TP0, FP0, _, FN = evalutation(filtered_anns, imageId2Anns[annsGroupId])
        TP1, FP1, ious, TN = evalutation(imageId2Anns[annsGroupId], filtered_anns)

        raw_results[imageId2Name[annsGroupId]] = {
            idx: [filtered_ann["category_id"], ious[idx]]
            for idx, filtered_ann in enumerate(filtered_anns)
        }
        if len(raw_results[imageId2Name[annsGroupId]]) == 7:
            raw_results[imageId2Name[annsGroupId]][7] = [np.nan, np.nan]

        raw_results[imageId2Name[annsGroupId]]["mean"] = np.nanmean(ious[0:8])
        total_ious.append(ious)

        assert np.sum(TP0) == np.sum(TP1), "not same as TP0, TP1"
        assert np.sum(FP0) == np.sum(FP1), "not same as FP0, FP1"

        if np.sum(TP1) == 8:
            sentence_correct += 1
        else:
            sentence_incorrect += 1

        total_TP += TP1[0:8]
        total_FP += FP1[0:8]
        total_TN += TN[0:8]
        total_FN += FN
        assert np.sum(TP1[8:10] + FP1[8:10] + TN[8:10]) == 0, "some annotation is error"

    print(f"not found : {np.sum(total_TN)}")
    print(f"found wrong : {np.sum(total_FN)}")
    print(
        f"precision : {np.sum(total_TP)/(np.sum(total_TP + total_FP + total_TN)) * 100}%"
    )
    print(
        f"recall : {np.sum(total_TP)/(np.sum(total_TP + total_FP) + np.sum(total_FN)) * 100}%"
    )
    print(
        f"accuracy : {np.sum(total_TP)/(np.sum(total_TP + total_FP + total_TN) + np.sum(total_FN)) * 100}%"
    )

    print(f"sentence correct count : {sentence_correct}")
    print(f"sentence incorrect count : {sentence_incorrect}")

    with open(f"{resultsPath}/result_raw.csv", "w") as cobj:
        csvwriter = csv.writer(cobj, delimiter=",")
        csvwriter.writerow(["filename"] + list(range(0, 8)) + ["total"])

        for imagename in imageId2Name.values():
            csvwriter.writerow(
                [imagename]
                + [
                    raw_result[0] - 1 if key != "mean" else round(raw_result, 2)
                    for key, raw_result in raw_results[imagename].items()
                ]
            )

    with open(f"{resultsPath}/result_summary.csv", "w") as cobj:
        csvwriter = csv.writer(cobj, delimiter=",")
        csvwriter.writerow(["category"] + list(range(0, 8)) + ["total"])
        csvwriter.writerow(["iou summary"])
        csvwriter.writerow(
            ["accuracy"]
            + [round(np.nanmean(np.array(total_ious)[:, i]), 2) for i in range(8)]
            + [round(np.nanmean(np.array(total_ious)[:, :8]), 2)]
        )
        csvwriter.writerow(
            ["std"]
            + [round(np.nanstd(np.array(total_ious)[:, i]), 2) for i in range(8)]
            + [round(np.nanstd(np.array(total_ious)[:, :8]), 2)]
        )

        csvwriter.writerow(["found summary"])
        csvwriter.writerow(["not found"] + total_TN.tolist() + [np.sum(total_TN)])
        csvwriter.writerow(
            ["found wrong"] + total_FN[0:8].tolist() + [np.sum(total_FN)]
        )
        csvwriter.writerow(["correct"] + total_TP.tolist() + [np.sum(total_TP)])
        csvwriter.writerow(["incorrect"] + total_FP.tolist() + [np.sum(total_FP)])
        csvwriter.writerow(
            ["total num"]
            + (total_TP + total_FP + total_TN).tolist()
            + [np.sum(total_TP + total_FP + total_TN)]
        )
        csvwriter.writerow(
            ["precision"]
            + (total_TP / (total_TP + total_FP + total_TN) * 100).tolist()
            + [np.sum(total_TP) / np.sum(total_TP + total_FP + total_TN) * 100]
        )
        csvwriter.writerow(
            ["recall"]
            + (total_TP / (total_TP + total_FP + total_FN[0:8]) * 100).tolist()
            + [
                np.sum(total_TP)
                / (np.sum(total_TP + total_FP) + np.sum(total_FN))
                * 100
            ]
        )
        csvwriter.writerow(
            ["accuracy"]
            + (
                total_TP / (total_TP + total_FP + total_TN + total_FN[0:8]) * 100
            ).tolist()
            + [
                np.sum(total_TP)
                / (np.sum(total_TP + total_FP + total_TN) + np.sum(total_FN))
                * 100
            ]
        )
        csvwriter.writerow(
            ["sentence correct"]
            + [sentence_correct]
            + ["sentence incorrect"]
            + [sentence_incorrect]
            + ["sentence accuracy"]
            + [sentence_correct / (sentence_incorrect + sentence_correct)]
        )
