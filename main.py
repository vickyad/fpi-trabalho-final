import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def grab_cut(img_target):
    mask = np.zeros(img_target.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, 450, 290)
    cv.grabCut(img_target, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img_target * mask2[:, :, np.newaxis]
    plt.imshow(img), plt.colorbar(), plt.show()


def poisson_editing(img_src, img_dst):
    im_mask = np.full(img_dst.shape, 255, dtype=np.uint8)  # isso que nós tem que fazer :D

    center = (round(img_src.shape[1] / 2), round(im_src.shape[0] / 2))

    im_clone = cv.seamlessClone(img_dst, img_src, im_mask, center, cv.MIXED_CLONE)

    cv.imshow("clone", im_clone)
    cv.waitKey(0)


if __name__ == '__main__':
    img = cv.imread('dog.jpg')
    im_src = cv.imread("images/airplane.jpg")
    im_dst = cv.imread("images/sky.jpg")

    grab_cut(img)
    poisson_editing(im_src, im_dst)
