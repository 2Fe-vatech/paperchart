import os
import json
import argparse

from tqdm import tqdm
from common import loadAnns, saveAnns


def isboxinbox(smallbox, bigbox, margin=0):
    return (
        smallbox[0] >= bigbox[0] - margin
        and (smallbox[0] + smallbox[2]) <= (bigbox[0] + bigbox[2]) + margin
        and smallbox[1] >= bigbox[1] - margin
        and (smallbox[1] + smallbox[3]) <= (bigbox[1] + bigbox[3]) + margin
    )


def bigger(x, y):
    return x if x > y else y


def smaller(x, y):
    return x if x < y else y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", help="base directory of images", required=True
    )
    parser.add_argument(
        "-t", "--target", help="data type [train, valid]", choices=["train", "valid"]
    )
    parser.add_argument(
        "-o", "--output", help="save result file path", default="results"
    )
    args = parser.parse_args()

    images, anns, classes = loadAnns(args.basedir, args.target)

    imgid2filename = {}
    for img in images:
        imgid2filename[img["id"]] = img["file_name"]

    dateboxid = 50000

    dateboxes = {}
    annid2ann = {}
    for idx, ann in enumerate(anns):
        if ann["image_id"] not in dateboxes.keys():
            dateboxes[ann["image_id"]] = {}

        if ann["category_id"] == 11:
            dateboxes[ann["image_id"]][dateboxid] = ann
            dateboxid += 1

        annid2ann[ann["id"]] = idx

    nboxesByDataId = {}

    for ann in anns:
        for idx in dateboxes[ann["image_id"]]:
            if isboxinbox(ann["bbox"], dateboxes[ann["image_id"]][idx]["bbox"]):
                if idx not in nboxesByDataId.keys():
                    nboxesByDataId[idx] = {}
                nboxesByDataId[idx][ann["id"]] = ann["bbox"]

    for bboxes in nboxesByDataId.values():
        flag = False
        temp_cat = {}
        temp_id = {}

        for idx, bbox in bboxes.items():
            if bbox[3] <= 12 and bboxes[idx + 1][3] <= 12 and flag == False:
                flag = True

                bbox_x = smaller(bbox[0], bboxes[idx + 1][0])
                bbox_y = smaller(bbox[1], bboxes[idx + 1][1])
                bbox_w = (
                    bigger(
                        (bboxes[idx + 1][0] + bboxes[idx + 1][2]), (bbox[0] + bbox[2])
                    )
                    - bbox_x
                )
                bbox_h = (
                    bigger(
                        (bboxes[idx + 1][1] + bboxes[idx + 1][3]), (bbox[1] + bbox[3])
                    )
                    - bbox_y
                )

                bbox = [bbox_x, bbox_y, bbox_w, bbox_h]
                anns[annid2ann[idx]]["bbox"] = bbox

            if flag == True:
                if bbox[3] <= 12:
                    temp_cat[idx + 1] = anns[annid2ann[idx + 1]]["category_id"]
                    temp_id[idx + 1] = anns[annid2ann[idx + 1]]["id"]

                    anns[annid2ann[idx + 1]]["category_id"] = anns[annid2ann[idx]][
                        "category_id"
                    ]
                    anns[annid2ann[idx + 1]]["id"] = anns[annid2ann[idx]]["id"]
                    del anns[annid2ann[idx]]

                    last_idx = list(bboxes.keys())[-1]
                    bbox = bboxes[last_idx]
                    anns.insert(
                        annid2ann[list(bboxes.keys())[-1]] - 1,
                        {
                            "iscrowd": 0,
                            "image_id": anns[annid2ann[idx]]["image_id"],
                            "bbox": [
                                bbox[0] + bbox[2] - 7.0,
                                bboxes[last_idx - 1][1],
                                7.0,
                                bboxes[last_idx - 1][3],
                            ],
                            "area": 7.0 * bboxes[last_idx - 1][3],
                            "category_id": 0,
                            "id": 0,
                        },
                    )

                elif bbox[2] < 20 and bbox_x != bbox[0]:
                    temp_cat[idx + 1] = anns[annid2ann[idx]]["category_id"]
                    anns[annid2ann[idx]]["category_id"] = temp_cat[idx]
                    temp_id[idx + 1] = anns[annid2ann[idx]]["id"]
                    anns[annid2ann[idx]]["id"] = temp_id[idx]

    errornum = 0
    erroridx = 0
    for bboxes in nboxesByDataId.values():
        flag = False

        for idx, bbox in bboxes.items():
            idx = idx + errornum
            if idx not in annid2ann.keys():
                annid2ann[idx] = annid2ann[idx - 1] + 1

            if idx + 1 not in annid2ann.keys():
                annid2ann[idx + 1] = annid2ann[idx] + 1

            if bbox[2] > 10 and bbox[2] < 20 and flag == False:
                flag = True
                erroridx = idx
                anns[annid2ann[idx]]["bbox"][2] = bbox[2] / 2
                anns.insert(
                    annid2ann[idx] + 1,
                    {
                        "iscrowd": 0,
                        "image_id": anns[annid2ann[idx]]["image_id"],
                        "bbox": [bbox[0] + bbox[2], bbox[1], bbox[2], bbox[3]],
                        "area": bbox[2] * bbox[3],
                        "category_id": anns[annid2ann[idx + 1]]["category_id"],
                        "id": anns[annid2ann[idx]]["id"],
                    },
                )
                errornum += 1

            if flag == True and bbox[2] < 20 and erroridx != idx:
                if idx - errornum + 1 != list(bboxes.keys())[-1]:
                    anns[annid2ann[idx]]["category_id"] = anns[annid2ann[idx + 1]][
                        "category_id"
                    ]
                else:
                    anns[annid2ann[idx]]["category_id"] = (
                        int(imgid2filename[anns[annid2ann[idx]]["image_id"]][9]) + 1
                    )

    for bboxes in nboxesByDataId.values():
        for idx, bbox in bboxes.items():
            if bbox[3] <= 12:
                anns[annid2ann[idx]]["bbox"][1] = bboxes[idx + 1][1]
                anns[annid2ann[idx]]["bbox"][3] = bboxes[idx + 1][3]

    saveAnns(
        os.path.join(args.output, "instances_{}.json".format(args.target)),
        {"images": images, "annotations": anns, "categories": classes},
    )
