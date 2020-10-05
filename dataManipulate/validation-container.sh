#!/bin/bash -i
# $1 : docker images tag
# $2 : model file name
# $3 : coef

dimgae=192.168.6.32:5000/paperchart:$1
docker run -it --rm --name ex_eval --entrypoint bash -d $dimgae
if [ $2 ]; then
    mc cp -r minio/models/paperchart/$2 ./

    if [ $? == 0 ]; then
        docker cp $2 ex_eval:/opt/weights/

        if [ $? == 0 ]; then
            docker exec -it ex_eval python coco_eval.py -p paperchart -c $3 -w /opt/weights/$2
            docker cp ex_eval:/opt/valid_bbox_results.json ./
            rm $2
        fi
    fi
fi
docker stop ex_eval


