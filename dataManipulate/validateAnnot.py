import os
import json
import argparse
from tqdm import tqdm
from common import getListCenter, seperate_by_date, seperate_by_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", help="base directory of images", required=True
    )
    parser.add_argument(
        "-t", "--target", help="data type [train, valid]", choices=["train", "valid"]
    )
    parser.add_argument(
        "-p", "--type", help="soft of data [crop, date]", choices=["crop", "date"]
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

    imageId2Name = {}
    for image in images:
        imageId2Name[image["id"]] = image["file_name"]

    catId2Name = {}
    for cat in classes:
        catId2Name[cat["id"]] = cat["name"]

    # overlapping image id check
    flag = False
    print("overlapping image id Check")
    for idx, image0 in enumerate(tqdm(images)):
        for image1 in images[idx + 1 :]:
            if image0["id"] == image1["id"]:
                flag = True
                print(
                    "overlap id : " + image0["id"],
                    "image name 0 : "
                    + imageId2Name[image0["id"]]
                    + "image name 1 : "
                    + imageId2Name[image1["id"]],
                )

    if flag == False:
        print("Nothing overlapping")

    # anns range in imageid check
    print("not exist image id Check")
    flag = False
    for ann in tqdm(anns):
        if ann["image_id"] not in imageId2Name.keys():
            flag = True
            print(
                "image name : "
                + imageId2Name[ann["image_id"]]
                + ", "
                + "ann id : "
                + ann["id"]
            )

    if flag == False:
        print("Nothing not exist image id")

    # overlapping ann id check
    flag = False
    print("overlapping category id Check")
    for idx, ann0 in enumerate(tqdm(anns)):
        for ann1 in anns[idx + 1 :]:
            if ann0["id"] == ann1["id"]:
                flag = True
                print(
                    f"overlap id : {ann0['id']} image name 0 : {imageId2Name[ann0['image_id']]}, image name 1 : {imageId2Name[ann1['image_id']]}"
                )

    if flag == False:
        print("Nothing overlapping")

    # number of annotations per date and image check
    print("number of annotations per date and image check")
    flag = False
    func = seperate_by_img if args.type == "crop" else seperate_by_date
    numberOfAnnsInImage = 1 if args.type == "crop" else 6

    for image_id, annsperimg in tqdm(func(anns).items()):
        if len(annsperimg) != numberOfAnnsInImage:
            flag = True
            print(
                f"number of dates : {len(annsperimg)}, image name : {imageId2Name[image_id]}"
            )

        for annid, annperimg in annsperimg.items():
            gtlabel = []

            for gtpart in (
                imageId2Name[image_id]
                .replace(" ", ".")
                .replace("_", ".")
                .split(".")[0:3]
            ):
                for gt in gtpart:
                    gtlabel.append(int(gt))

            if len(annperimg["subbox"]) != 8:
                flag = True
                print(
                    f"number of numbers : {str(len(annperimg['subbox']))}, image name : {imageId2Name[image_id]}, annotation id : {annid}"
                )

            # real label and annot label different check
            legacy_x = -100
            for idx, ann in enumerate(annperimg["subbox"]):
                cat_id, bbox = list(ann.items())[0]

                if len(gtlabel) <= idx:
                    print(
                        f"[category error] ann cat : {cat_id - 1}, image name : {imageId2Name[image_id]}, annotation id : {annid}"
                    )
                elif cat_id - 1 != gtlabel[idx]:
                    print(
                        f"[category error] ann cat : {cat_id - 1}, cat : {gtlabel[idx]}, image name : {imageId2Name[image_id]}, annotation id : {annid}"
                    )

                if bbox[3] < 14.0:
                    print(
                        f"[height error] height : {bbox[3]}, image name : {imageId2Name[image_id]}, annotation id : {annid}"
                    )

                if abs(legacy_x - bbox[0]) <= 2:
                    print(
                        f"[height error] x_coord0 : {legacy_x}, x_coord1 : {bbox[0]}, image name : {imageId2Name[image_id]}, annotation id : {annid}"
                    )

                legacy_x = bbox[0]

    if flag == False:
        print("Nothing in annotations")
