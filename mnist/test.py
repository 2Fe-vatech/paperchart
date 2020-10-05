import cv2
import math
import random
import numpy as np
from PIL import Image

img = cv2.imread("datasets/numbers/train/0/train30090_0.png")
# for i in range(10):
#     d = np.random.randint(0, 5, 4)
#     new_shape = [img.shape[0] + d[0] + d[2], img.shape[1] + d[1] + d[3], img.shape[2]]
#     newimg = np.ones((new_shape)) * 255
#     newimg[d[0]:img.shape[0] + d[0], d[1]:img.shape[1] + d[1]] = img
#     cv2.imwrite(f'temp{i}.jpg', newimg)

for i in range(10):
    angle = 45
    rotate = random.randint(-angle, angle)
    size = (32, 32)
    assert len(size) == 2, "variable size form should be (widht, height)"

    padding = abs(img.shape[0] - img.shape[1]) // 2
    if img.shape[0] > img.shape[1]:
        newimg = np.ones((img.shape[0], img.shape[0], img.shape[2])) * 255
        newimg[:, padding : padding + img.shape[1], :] = img
    else:
        newimg = np.ones((img.shape[1], img.shape[1], img.shape[2])) * 255
        newimg[padding : padding + img.shape[0], :, :] = img

    newimg = Image.fromarray(np.array(newimg, dtype="uint8"))
    newimg = newimg.rotate(rotate, fillcolor=((255, 255, 255)))
    newimg = np.asarray(newimg, dtype=float)

    select_crop = random.randint(0, 4)
    width_edge, height_edge = int(newimg.shape[0] * 0.2), int(newimg.shape[1] * 0.2)

    if select_crop == 0:
        newimg = newimg[
            0 : newimg.shape[1] - height_edge, 0 : newimg.shape[0] - width_edge, :
        ]
    elif select_crop == 1:
        newimg = newimg[height_edge:, 0 : newimg.shape[0] - width_edge, :]
    elif select_crop == 2:
        newimg = newimg[0 : newimg.shape[1] - height_edge :, width_edge:, :]
    elif select_crop == 3:
        newimg = newimg[height_edge:, width_edge:, :]
    elif select_crop == 4:
        newimg = newimg[
            height_edge // 2 : newimg.shape[1] + height_edge // 2,
            width_edge // 2 : newimg.shape[0] + width_edge // 2,
            :,
        ]
    newimg = cv2.resize(newimg, size)
    print(i, select_crop)

    cv2.imwrite(f"temp{i}.jpg", newimg)
