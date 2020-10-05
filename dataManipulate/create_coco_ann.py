import os
import sys
import json
import argparse
from tqdm import tqdm
from label_detector import date_labeling


def paperchar2coco(basedir, target="train", supercategory="paperchart", basecategory=1):

    basedir = os.path.abspath(os.path.expanduser(basedir))
    assert os.path.isdir(basedir), basedir + " is not exist"
    os.makedirs(os.path.join(basedir, "annotations"), exist_ok=True)

    old = sys.stdout
    sys.stdout = open(
        os.path.join(basedir, "annotations", "instances_{}.json".format(target)), "w"
    )

    classes = {}

    with open(os.path.join(basedir, "classes.json")) as fobj:
        classes.update(json.load(fobj))
        assert len(classes) != 0, "classes has no data"

    imageids = 10000
    annotationids = 30000

    images = []
    annotations = []
    categories = []
    label2id = {}

    for idxclass, oneclass in enumerate(sorted(classes)):
        categories.append(
            {
                "supercategory": supercategory,
                "id": idxclass + basecategory,
                "name": oneclass.rstrip(),
            }
        )
        label2id[oneclass.rstrip()] = idxclass + basecategory

    imagespath = os.path.join(basedir, target)
    assert os.path.isdir(imagespath), imagespath + " is not exist"

    for filename in tqdm(os.listdir(imagespath)):
        groups, img_shape = date_labeling(os.path.join(imagespath, filename), pad=0)
        imageids = imageids + 1
        images.append(
            {
                "file_name": filename,
                "height": img_shape[0],
                "width": img_shape[1],
                "id": imageids,
            }
        )

        for anns in groups.values():
            for ann in anns:
                label = list(ann.keys())[0]
                bbox = list(ann.values())[0]
                annotationids = annotationids + 1
                annotations.append(
                    {
                        "iscrowd": 0,
                        "image_id": imageids,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "category_id": label2id[label],
                        "id": annotationids,
                    }
                )

    assert len(images) != 0 and len(annotations) != 0 and len(categories) != 0, (
        os.path.join(basedir, target) + " has no data to write"
    )

    print(
        json.dumps(
            {"images": images, "annotations": annotations, "categories": categories},
            indent=2,
        )
    )

    sys.stdout = old


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--basedir", help="base directory", required=True)
    parser.add_argument(
        "-t",
        "--target",
        help="[train, valid]",
        choices=["train", "valid"],
        default="train",
    )
    args = parser.parse_args()

    paperchar2coco(
        args.basedir, target=args.target, supercategory="paperchart", basecategory=1
    )
