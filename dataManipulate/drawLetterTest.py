import cv2
import numpy as np

img = np.ones((1024, 1024, 3)) * 255

for i in range(0, 10):
    cv2.putText(img, str(i), (i * 20 + 300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

for i in range(0, 10):
    cv2.putText(img, str(i), (i * 20 + 300, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

for i in range(0, 10):
    cv2.putText(img, str(i), (i * 20 + 300, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)

for i in range(0, 10):
    cv2.putText(img, str(i), (i * 20 + 300, 400), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

for i in range(0, 10):
    cv2.putText(img, str(i), (i * 20 + 300, 500), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)

for i in range(0, 10):
    cv2.putText(img, str(i), (i * 20 + 300, 600), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 0), 2)

for i in range(0, 10):
    cv2.putText(img, str(i), (i * 20 + 300, 700), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 0, 0), 2)

cv2.imwrite("test.png", img)