#!/bin/bash

docker run -it --name mnist -v $PWD:/opt -w /opt -p 9909:9909 --gpus all $@ 192.168.6.32:5000/pytorch:py36-cu101
