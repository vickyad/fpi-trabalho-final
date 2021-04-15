import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def grab_cut(img_target):
    mask = np.zeros(img_target.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (0, 0, img_target.shape[0], img_target.shape[1])
    cv.grabCut(img_target, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img_target * mask2[:, :, np.newaxis]
    plt.imshow(img), plt.colorbar(), plt.show()


def poisson_editing(img_src, img_dst):
    mask = 255 * np.ones(img_dst.shape, img_dst.dtype)

    width, height, channels = im_src.shape
    center = (round(height / 2), round(width / 2))

    mixed_clone = cv.seamlessClone(img_dst, img_src, mask, center, cv.MIXED_CLONE)

    cv.imwrite("images/opencv-mixed-clone-example.jpg", mixed_clone)


if __name__ == '__main__':
    img = cv.imread('dog.jpg')
    img_cut = cv.imread("images/airplane_cut.jpg")
    im_src = cv.imread("images/airplane.jpg")
    im_dst = cv.imread("images/sky.jpg")

    grab_cut(img_cut)
    poisson_editing(im_src, im_dst)
