#!/bin/bash -i
# $1 : base directory

rm results/*

set -e
cp $1/annotations/instances_train.date.json $1/annotations/instances_train.json
cp $1/annotations/instances_valid.date.json $1/annotations/instances_valid.json
python eval_date.py -b $1 -t valid -p valid_bbox_results.json
python3 figures.py -b $1 -t valid -p valid_bbox_results.json -v -l 0.1
cp $1/annotations/instances_train.all.json $1/annotations/instances_train.json
cp $1/annotations/instances_valid.all.json $1/annotations/instances_valid.json