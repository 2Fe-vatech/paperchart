import os
import json
import argparse

from tqdm import tqdm


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
    classes = jdata["categories"]

    annotationids = 30000

    for ann in tqdm(anns):
        annotationids += 1
        ann["id"] = annotationids
        ann["area"] = float(ann["bbox"][2]) * float(ann["bbox"][3])

        if "ares" in ann.keys():
            del ann["ares"]

    annids = []

    for ann in tqdm(anns):
        assert ann["id"] not in annids, ann["id"] + " exist two more"
        annids.append(ann["id"])
    assert len(annids) == len(anns), "error "

    os.makedirs(args.output, exist_ok=True)

    with open(
        os.path.join(args.output, "instances_{}.json".format(args.target)), "w"
    ) as jobj:
        json.dump(
            {"images": images, "annotations": anns, "categories": classes},
            jobj,
            indent=2,
        )
