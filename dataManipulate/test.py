# import cv2


# def display_image(name, image):
#     showimages = cv2.resize(image, (200, 100))
#     cv2.imshow(name, showimages)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# image = cv2.imread("croppedimages/train/2020.01.04_2.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# ret, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
# image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# display_image("test", image)

import csv
import numpy as np

results = np.zeros((10, 10))

with open("tests.csv", "w") as cobj:
    csvwirter = csv.writer(cobj, delimiter=" ")
    csvwirter.writerow(["filename", "GT", "result"])
    csvwirter.writerows(results)
