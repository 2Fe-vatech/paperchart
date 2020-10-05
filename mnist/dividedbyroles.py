import os
import argparse
from shutil import copy2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", help="base directory of images", required=True
    )
    parser.add_argument(
        "-t", "--target", help="data type [train, valid]", choices=["train", "valid"]
    )
    # parser.add_argument("--rate", help="train and valid data num rate", type=float, default=0.8)
    parser.add_argument(
        "-r", "--results", help="results path", default="datasets/numbers"
    )
    args = parser.parse_args()

    basedir = os.path.abspath(
        os.path.expanduser(os.path.join(args.basedir, args.target))
    )
    assert os.path.isdir(basedir), basedir + " is not file or exists"

    numimages = {}
    min_amount_data = 100000
    for num in range(10):
        numimages[num] = [
            fn
            for fn in os.listdir(os.path.join(basedir, str(num)))
            if fn.endswith(".png")
        ]

        if min_amount_data > len(numimages[num]):
            min_amount_data = len(numimages[num])

    for label, images in numimages.items():
        os.makedirs(os.path.join(args.results, args.target, str(label)), exist_ok=True)

        for imagename in images[:min_amount_data]:
            copy2(
                os.path.join(basedir, str(label), imagename),
                os.path.join(args.results, args.target, str(label), imagename),
            )
