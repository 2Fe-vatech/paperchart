import os
import json
import cv2
import numpy as np


def display_image(name, image, size=(600, 800)):
    showimages = cv2.resize(image, size)
    cv2.imshow(name, showimages)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def loadAnns(basedir, target):
    basedir = os.path.abspath(os.path.expanduser(basedir))
    assert os.path.isdir(basedir), basedir + " is not file or exists"

    annsfile = os.path.join(basedir, "annotations", "instances_{}.json".format(target))
    assert os.path.isfile(annsfile), annsfile + " is not file or exists"

    with open(annsfile, "r") as jobj:
        jdata = json.load(jobj)
        assert (
            "images" in jdata and "annotations" in jdata and "categories" in jdata
        ), "[annotation file error] illegal format"

    return jdata["images"], jdata["annotations"], jdata["categories"]


def saveAnns(filename, data):
    with open(filename, "w") as jobj:
        json.dump(data, jobj, indent=3)


def cal_iou(bbox1, bbox2):
    overlap_x_start = bbox1[0] if bbox1[0] >= bbox2[0] else bbox2[0]
    overlap_y_start = bbox1[1] if bbox1[1] >= bbox2[1] else bbox2[1]

    overlap_x_end = (
        (bbox2[0] + bbox2[2])
        if (bbox1[0] + bbox1[2]) >= (bbox2[0] + bbox2[2])
        else (bbox1[0] + bbox1[2])
    )
    overlap_y_end = (
        (bbox2[1] + bbox2[3])
        if (bbox1[1] + bbox1[3]) >= (bbox2[1] + bbox2[3])
        else (bbox1[1] + bbox1[3])
    )

    overlap_w = overlap_x_end - overlap_x_start
    overlap_h = overlap_y_end - overlap_y_start

    if overlap_w < 0 or overlap_h < 0:
        return -1

    all_area = bbox1[2] * bbox1[3] + bbox2[2] * bbox2[3] - overlap_w * overlap_h
    overlap_area = overlap_w * overlap_h
    assert all_area != 0, "all area could not be zero"

    return overlap_area / all_area


def getimageinfo(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w, c = image.shape
    area = w * h
    return image, w, h, area


def filter_by_iou(annsGroup):
    filtered_anns = []
    for idx, ann0 in enumerate(annsGroup):

        flag = False

        for ann1 in annsGroup:
            if (
                ann0 != ann1
                and ann0["score"] < ann1["score"]
                and cal_iou(ann0["bbox"], ann1["bbox"]) >= 0.3
            ):
                flag = True
                temp = cal_iou(ann0["bbox"], ann1["bbox"])

        if flag == False:
            filtered_anns.append(ann0)

    return filtered_anns


def getListCenter(bboxes):
    npbboxes = np.array(bboxes)
    bboxescenter = np.zeros((len(npbboxes), 2))
    bboxescenter[:, 0] = npbboxes[:, 0] + npbboxes[:, 2] / 2
    bboxescenter[:, 1] = npbboxes[:, 1] + npbboxes[:, 3] / 2

    return bboxescenter


def seperate_by_date(anns):
    dateboxlist = {}
    bboxidxlist = {}
    bboxlist = {}

    id2catid = {}
    for ann in anns:
        id2catid[ann["id"]] = ann["category_id"]

    for ann in anns:
        if ann["category_id"] == 11:
            if ann["image_id"] not in dateboxlist.keys():
                dateboxlist[ann["image_id"]] = {}
            dateboxlist[ann["image_id"]][ann["id"]] = {"bbox": ann["bbox"]}
        else:
            if ann["image_id"] not in bboxlist.keys():
                bboxlist[ann["image_id"]] = []
                bboxidxlist[ann["image_id"]] = []
            bboxidxlist[ann["image_id"]].append(ann["id"])
            bboxlist[ann["image_id"]].append(ann["bbox"])

    for imageid, datebox in dateboxlist.items():
        bboxcenter = getListCenter(bboxlist[imageid])
        for idx, db in datebox.items():
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
    return dateboxlist


def seperate_by_img(anns):

    dateboxlist = {}

    id2catid = {}
    for ann in anns:
        id2catid[ann["id"]] = ann["category_id"]

    imageid2anns = {}
    for ann in anns:
        if ann["image_id"] not in imageid2anns:
            imageid2anns[ann["image_id"]] = []
        imageid2anns[ann["image_id"]].append(ann)

    for imageid, annsperimage in imageid2anns.items():

        if imageid not in dateboxlist:
            dateboxlist[imageid] = {}

        annid = annsperimage[0]["id"]
        if annid not in dateboxlist[imageid]:
            dateboxlist[imageid][annid] = {}

        subbox = []
        for ann in annsperimage:
            subbox.append({id2catid[ann["id"]]: ann["bbox"]})

        dateboxlist[imageid][annid]["subbox"] = subbox

    return dateboxlist
