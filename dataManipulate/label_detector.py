import cv2
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
from common import display_image


def get_border_from_hull(hull, pad=0):
    hull = np.array(hull)
    x_max, x_min = np.max(hull[..., 0]), np.min(hull[..., 0])
    y_max, y_min = np.max(hull[..., 1]), np.min(hull[..., 1])
    return x_max + pad, x_min - pad, y_max + pad, y_min - pad


def get_center_from_hull(hull):
    x_max, x_min, y_max, y_min = get_border_from_hull(hull)
    return ((x_max + x_min) / 2, (y_max + y_min) / 2)


def get_distance_from_hulls(hull0, hull1):
    h0 = get_center_from_hull(hull0)
    h1 = get_center_from_hull(hull1)
    return np.linalg.norm(np.array(h1) - np.array(h0))


def create_center_array(hulls):
    hulls = np.array(hulls)
    temp = np.zeros((hulls.shape[0], 2))
    temp[:, 0] = (hulls[:, 0, 0, 0] + hulls[:, 1, 0, 0] + 0.5) // 2
    temp[:, 1] = (hulls[:, 0, 0, 1] + hulls[:, 2, 0, 1] + 0.5) // 2
    return temp


def cal_ratio(hull, pad=0):
    x_max, x_min, y_max, y_min = get_border_from_hull(hull, pad=pad)
    width = x_max - x_min
    height = y_max - y_min
    ratio = height / width
    return x_max, x_min, y_max, y_min, ratio


def date_labeling(path, outfile="results", show=False, save=False, pad=20):
    assert os.path.isfile(path), path + " is not exsit"
    assert os.path.splitext(path)[-1] in [".jpg", ".png", ".bmp", ".JPEG", ".jpeg"], (
        path + " is not picture file"
    )
    pad = 0 if pad < 0 else pad

    orig_images = cv2.imread(path)
    crop_images = orig_images[:, 0:-10, :]
    gray_images = cv2.cvtColor(crop_images, cv2.COLOR_BGR2GRAY)
    img_binary = cv2.adaptiveThreshold(
        gray_images, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 40
    )

    # if save:
    #     cv2.imwrite(outfile + ".png", img_binary)

    contours, hierarchy = cv2.findContours(
        img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    hulls = []

    for idx, cnt in enumerate(contours):
        hull = cv2.convexHull(cnt)
        hull_hierarchy = hierarchy[0][idx]
        hulls_area = cv2.contourArea(hull)

        if hull_hierarchy[3] == 0 and hulls_area > 20 and hulls_area < 5000:
            x_max, x_min, y_max, y_min, ratio = cal_ratio(hull)
            if ratio > 0.9 or hulls_area > 500:
                hulls.append(
                    np.array(
                        [
                            [[x_max + pad, y_max + pad]],
                            [[x_min - pad, y_max + pad]],
                            [[x_min - pad, y_min - pad]],
                            [[x_max + pad, y_min - pad]],
                        ]
                    )
                )

    # for hull in hulls:
    #     cv2.drawContours(orig_images, [hull], 0, (0, 0, 255), 1)

    # if save:
    #     cv2.imwrite(outfile + ".png", orig_images)

    hull_groups = {}
    group_idx = 0

    while len(hulls) != 0:
        hull_groups[group_idx] = []
        hull_groups[group_idx].append(hulls.pop(0))

        isinLinst = True
        while isinLinst == True:
            inListIdx = []
            for ghull in hull_groups[group_idx]:
                if len(hulls) != 0:
                    hulls_center = create_center_array(hulls)
                    ghull_center = get_center_from_hull(ghull)
                    sub_center = hulls_center - ghull_center
                    distance_array = (
                        sub_center[:, 0] ** 2 + sub_center[:, 1] ** 2
                    ) ** 0.5
                    inListIdx = np.where(distance_array < 100)[0].tolist()

                    if len(inListIdx) == 0:
                        isinLinst = False
                    else:
                        for idx in sorted(inListIdx, reverse=True):
                            hull_groups[group_idx].append(hulls.pop(idx))
                else:
                    isinLinst = False

        group_idx += 1

    results = {}

    filename = os.path.splitext(os.path.split(path)[-1])[0]
    label = [lb for lb in filename if lb != "."]

    for idx, hulls in hull_groups.items():
        if idx not in list(results.keys()):
            results[idx] = []

        hulls = np.array(hulls)
        x_mins = hulls[:, 1, :, 0].flatten()
        sorted_index = x_mins.argsort()
        hull_groups[idx] = hulls[sorted_index, ...]

        for num, hull in enumerate(hull_groups[idx]):
            x_max, x_min, y_max, y_min = get_border_from_hull(hull)

            if len(label) > num:
                results[idx].append(
                    {
                        label[num]: [
                            float(x_min),
                            float(y_min),
                            float(x_max - x_min),
                            float(y_max - y_min),
                        ]
                    }
                )

            if save or show:
                # if len(label) > num and label[num] == '0':
                #     cv2.putText(orig_images, '{}, {}'.format(x_min, y_min), (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

                # cv2.putText(
                #     orig_images,
                #     label[num] if len(label) > num else str(num),
                #     (int(x_min), int(y_min)),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     1,
                #     (0, 0, 0),
                #     1,
                # )
                # cv2.drawContours(orig_images, hull, 0, (0, 0, 255), 3)
                cv2.rectangle(
                    orig_images, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1
                )

        x_max, x_min, y_max, y_min = get_border_from_hull(hulls)
        new_hull = np.array(
            [[[x_max, y_max]], [[x_min, y_max]], [[x_min, y_min]], [[x_max, y_min]]]
        )

        if cv2.contourArea(new_hull) > 500:
            results[idx].append(
                {
                    "date": [
                        float(x_min),
                        float(y_min),
                        float(x_max - x_min),
                        float(y_max - y_min),
                    ]
                }
            )

            if save or show:
                cv2.drawContours(orig_images, [new_hull], 0, (0, 0, 255), 1)

    if save:
        cv2.imwrite(outfile + ".png", orig_images)
        # with open(outfile + ".json", "w") as jobj:
        #     json.dump(results, jobj, indent=3)

    if show:
        display_image(path, orig_images)

    return results, orig_images.shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", help="base directory of images", required=True
    )
    parser.add_argument(
        "-t",
        "--target",
        help="[train, valid]",
        choices=["train", "valid"],
        default="train",
    )
    parser.add_argument(
        "-p", "--padding", help="padding on annotations", type=int, default=0
    )
    parser.add_argument("-w", "--show", help="show result image", action="store_true")
    parser.add_argument(
        "-v", "--save", help="save result image and json file", action="store_true"
    )
    args = parser.parse_args()

    basedir = os.path.abspath(
        os.path.expanduser(os.path.join(args.basedir, args.target))
    )
    assert os.path.isdir(basedir), basedir + " is not exist"

    for imagefile in tqdm(os.listdir(basedir)):
        if os.path.splitext(imagefile)[-1] in [".jpg", ".png", ".bmp"]:
            date_labeling(
                os.path.join(basedir, imagefile),
                outfile=os.path.join("results", imagefile),
                show=args.show,
                save=args.save,
                pad=args.padding,
            )
