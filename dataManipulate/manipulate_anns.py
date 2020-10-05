import os
import json
import argparse

from tqdm import tqdm
from common import loadAnns, saveAnns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", help="base directory of images", required=True
    )
    parser.add_argument(
        "-t", "--target", help="data type [train, valid]", choices=["train", "valid"]
    )
    parser.add_argument(
        "-c",
        "--category",
        help="category type [date, number, all]",
        choices=["date", "number", "all"],
        default="all",
    )
    parser.add_argument(
        "-o", "--output", help="save result file path", default="results"
    )
    args = parser.parse_args()

    images, anns, classes = loadAnns(args.basedir, args.target)

    assert len(classes) == 11, "all data annotation set should be needed"
    if args.category == "date":
        allowcatnum = [11]
        classes = [classes[10]]
        classes[0]["id"] = 1
    elif args.category == "number":
        allowcatnum = list(range(1, 11))
        classes = classes[0:10]

    new_anns = []
    if args.category != "all":
        for ann in anns:
            if ann["category_id"] in allowcatnum:
                if args.category == "date":
                    ann["category_id"] = 1
                new_anns.append(ann)

    os.makedirs(args.output, exist_ok=True)

    saveAnns(
        os.path.join(args.output, "instances_{}.json".format(args.target)),
        {"images": images, "annotations": new_anns, "categories": classes},
    )
