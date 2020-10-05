import json
import os

# filelist = [
#     '2019.03.04_9.jpg',
#     '2019.06.01_14.jpg',
#     '2019.10.02_10.jpg',
#     '2019.10.30_16.jpg'
# ]

filelist = []
with open("errorimage.txt") as tobj:
    lines = tobj.readlines()
    filelist = list([line.replace("\n", ".jpg") for line in lines])
    filelist[-1] = filelist[-1] + ".jpg"

with open("croppedimages/annotations/instances_train.json") as jobj:
    jdata = json.load(jobj)

fileidxlist = []
for idx, image in enumerate(jdata["images"]):

    if image["file_name"] in filelist:
        fileidxlist.append(image["id"])
        del jdata["images"][idx]

for idx, ann in enumerate(jdata["annotations"]):
    if ann["image_id"] in fileidxlist:
        del jdata["annotations"][idx]

with open("instances_train.json", "w") as jobj:
    json.dump(jdata, jobj, indent=2)
