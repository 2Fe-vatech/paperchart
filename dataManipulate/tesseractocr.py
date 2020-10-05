import os
import csv
import argparse
import pytesseract
from PIL import Image
from tqdm import tqdm
from Levenshtein import distance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--basedir", help="base directory of images", required=True
    )
    parser.add_argument(
        "-t", "--target", help="data type [train, valid]", choices=["train", "valid"]
    )
    parser.add_argument("--psm", help="psm option of tesseract orc", type=int)
    parser.add_argument("--oem", help="psm option of tesseract orc", choices=["1", "3"])
    parser.add_argument(
        "-s",
        "--similar",
        help="include result which not same but similar",
        action="store_true",
    )
    parser.add_argument(
        "-o", "--output", help="save result file path", default="results"
    )
    args = parser.parse_args()

    basedir = os.path.abspath(os.path.expanduser(args.basedir))
    assert os.path.isdir(basedir), basedir + " is not a directory"
    assert os.path.isdir(os.path.join(basedir, args.target)), (
        os.path.join(basedir, args.target) + " is not a directory"
    )

    imglist = [
        fn
        for fn in os.listdir(os.path.join(basedir, args.target))
        if os.path.splitext(fn)[-1] in [".bmp", ".jpg", ".BMP"]
    ]
    assert len(imglist) != 0, (
        os.path.join(basedir, args.target) + " should have at least one more images"
    )

    countMatch = 0
    countBlank = 0
    if args.similar:
        countSimilar = 0

    pytesseractCfg = ""
    desc = ""
    if args.oem:
        pytesseractCfg += " --oem " + args.oem
        desc += "oem : " + args.oem

    if args.psm:
        pytesseractCfg += " --psm " + str(args.psm)
        desc += " psm : " + str(args.psm)

    results = []
    for filename in tqdm(imglist, desc=desc):
        resultString = pytesseract.image_to_string(
            Image.open(os.path.join(basedir, args.target, filename)),
            config=pytesseractCfg,
        )

        sumString = ""
        for char in list(filter(str.isdigit, resultString)):
            sumString += char
        gt = filename.split("_")[0].replace(".", "")

        result = [filename, gt, sumString]

        if sumString == gt:
            countMatch += 1
            result.append("match")
        elif len(sumString) == 0:
            countBlank += 1
            result.append("blank")
        else:
            result.append("unmatch")

        if args.similar and distance(sumString, gt) <= 2:
            if result[-1] == "unmatch":
                result[-1] = "similar"
                countSimilar += 1

        results.append(result)

    with open(
        os.path.join(args.output, "oem{}psm{}.csv".format(args.oem, args.psm)), "w"
    ) as cobj:
        csvwirter = csv.writer(cobj, delimiter=",")
        csvwirter.writerow(["filename", "GT", "result", "adjudgement"])
        csvwirter.writerows(results)

        row = ["match", countMatch, "Blank", countBlank]
        if args.similar:
            row.extend(["similar", countSimilar])
        csvwirter.writerow(row)

        row = ["accuracy(match)", "{}%".format(countMatch / len(imglist) * 100)]
        row.extend(["Blank", "{}%".format(countBlank / len(imglist) * 100)])
        if args.similar:
            row.extend(
                [
                    "accuracy(similar)",
                    "{}%".format((countBlank + countSimilar) / len(imglist) * 100),
                ]
            )
        csvwirter.writerow(row)

    print("accuracy : {}%".format(countMatch / len(imglist) * 100))
    print("Blank : {}%".format(countBlank / len(imglist) * 100))
    print(
        "accuracy(similar) : {}%".format(
            (countMatch + countSimilar) / len(imglist) * 100
        )
    )
