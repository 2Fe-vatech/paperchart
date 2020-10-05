#!/bin/bash -i
for oem in 1 3
do
    for psm in 1 3 4 5 6 7 8 9 10 11 12 13
    do
        # echo "oem : $oem psm : $psm"
        python tesseractocr.py -b datasets/paperchart2_c -t valid --oem $oem --psm $psm -s
    done
done
