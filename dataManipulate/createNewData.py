import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from common import loadAnns, saveAnns
from fixannot import isboxinbox

def copydateimg(img, srcbbox, destbbox, move):

    def cropimg(img, bbox):
        bbox = np.array(bbox)[0]
        bbox = np.array(bbox + 0.5, dtype=int)
        return img[bbox[1]:bbox[1]+bbox[3] + 3,bbox[0]:bbox[0]+bbox[2] + 10, :]

    cimg = cropimg(img, [srcbbox])
    new_bbox = [int(destbbox[0]), int(destbbox[1] + 0.5) + move, srcbbox[2], srcbbox[3]]
    img[new_bbox[1]:new_bbox[1] + cimg.shape[0], new_bbox[0]:new_bbox[0] + cimg.shape[1],:] = cimg

    return img, new_bbox

def copydateimg_with_ann(img, srcann, destann, move):
    img, new_bbox = copydateimg(img, srcann["bbox"], destann["bbox"], move=move)

    new_dateann = deepcopy(srcann)

    x_move = srcann["bbox"][0] - new_bbox[0]
    y_move = srcann["bbox"][1] - new_bbox[1]
    for numann in new_dateann["child"]:
        numann["bbox"][0] -= x_move
        numann["bbox"][1] -= y_move
        # bbox = np.array(np.array(numann["bbox"]) + 0.5, dtype=int)
        # img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,0,0), 1)
    
    new_dateann["bbox"] = new_bbox

    # img = cv2.rectangle(img, (new_bbox[0], new_bbox[1]), (int(new_bbox[0]+new_bbox[2] + 0.5),int(new_bbox[1]+new_bbox[3]+0.5)),(0,0,0), 1)

    return img, new_dateann

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-b", "--basedir", help="base directory", required=True)
    parse.add_argument("-t", "--target", choices=["train", "valid"])
    parse.add_argument("-o", "--outdir", help="result save directory", default="results")
    parse.add_argument("-g", "--gap", help="gap between augmented dateboxes", type=int, default=25)
    args = parse.parse_args()

    datapath = os.path.abspath(os.path.expanduser(args.basedir))
    copymethod = []

    for m in [(0,3,3,6), (3,6,0,3)]:
        for j in range(m[0], m[1]):
            move = -args.gap * 3

            for k in range(0,2):
                for i in range(m[2], m[3]):
                    copymethod.append([i, j, move])
                    
                    if move == -args.gap:
                        move += args.gap * 2
                    else:
                        move += args.gap

    os.makedirs(os.path.join(args.outdir, args.target), exist_ok=True)

    images, annotations, categories = loadAnns(datapath, args.target)

    imageid2ann = {}
    for ann in annotations:
        if ann["image_id"] not in imageid2ann:
            imageid2ann[ann["image_id"]] = []
        imageid2ann[ann["image_id"]].append(ann)

    ''' 
    sorted datebox 
    -----------------
    |               |
    |  0        3   |
    |               |
    |  1        4   |
    |               |
    |  2        5   |
    |               |
    -----------------
    '''

    new_dateanns = []
    for image in tqdm(images, desc="create images : "):
        img = cv2.imread(os.path.join(datapath, args.target, image["file_name"]))

        dateanns = []
        for ann in imageid2ann[image["id"]]:
            if ann["bbox"][2] > 30:
                dateanns.append(ann)
                new_dateanns.append(ann)
            
        dateboxes = np.zeros((6,4))

        for idx, dateann in enumerate(dateanns):
            dateboxes[idx] = dateann["bbox"]
        sortedidx = np.argsort(dateboxes[:, 0])
        
        dateboxes = dateboxes[sortedidx]
        sortedidx2 = np.zeros(6, dtype=int)
        sortedidx2[:3] = np.argsort(dateboxes[:3, 1])
        sortedidx2[3:] = np.argsort(dateboxes[3:, 1]) + 3

        mapidx = [sortedidx[sidx] for idx, sidx in enumerate(sortedidx2)]
        dateanns = [dateanns[idx] for idx in mapidx]
        
        for dateann in dateanns:
            dateann["child"] = []

            for ann in imageid2ann[image["id"]]:
                if dateann["id"] != ann["id"] and isboxinbox(ann["bbox"], dateann["bbox"], 2):
                    dateann["child"].append(ann)

        for method in copymethod:
            img, new_dateann = copydateimg_with_ann(img, dateanns[method[0]], dateanns[method[1]], method[2])
            new_dateanns.append(new_dateann)

        cv2.imwrite(os.path.join(args.outdir, args.target, image["file_name"]), img)
    
    anns = []
    annidx = 30000
    for dateann in new_dateanns:
        for ann in dateann["child"]:
            annidx += 1
            ann["id"] = annidx
            anns.append(ann)
        
        del dateann["child"]
        anns.append(dateann)
    
    os.makedirs(os.path.join(args.outdir, "annotations"), exist_ok=True)

    saveAnns(
        os.path.join(args.outdir, f"annotations/instances_{args.target}.json"), 
        {"images": images, "annotations": anns, "categories": categories}
    )
    

