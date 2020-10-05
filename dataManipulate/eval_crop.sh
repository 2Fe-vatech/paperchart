#!/bin/bash -i
# $1 : base directory
# $2 : threshold value

rm results/*

set -e
python eval_crop.py -b $1 -t valid -p valid_bbox_results.json
python3 figures.py -b $1 -t valid -p valid_bbox_results.json -v -l 0.3
