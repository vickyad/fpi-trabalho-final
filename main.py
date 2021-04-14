import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('dog.jpg')
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (50, 50, 450, 290)
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img * mask2[:, :, np.newaxis]
plt.imshow(img), plt.colorbar(), plt.show()


im_src = cv.imread("images/airplane.jpg")
im_dst = cv.imread("images/sky.jpg")
im_mask = np.full(im_dst.shape, 255, dtype=np.uint8)    # isso que nós tem que fazer :D

center = (round(im_src.shape[1] / 2), round(im_src.shape[0] / 2))

im_clone = cv.seamlessClone(im_dst, im_src, im_mask, center, cv.MIXED_CLONE)

cv.imshow("clone", im_clone)
cv.waitKey(0)
