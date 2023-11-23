import os
import cv2
import numpy as np


def division_confidence_by_threshold(thrs_high=192, thrs_low=64):
    srcdata_path = './datasets/crack500/train/mask_CAM'
    result_confidence_high_path = './datasets/crack500/train/mask_CAM_threshold_confidene_high'
    result_confidence_low_path = './datasets/crack500/train/mask_CAM_threshold_confidence_low'
    if not os.path.exists(result_confidence_high_path):
        os.mkdir(result_confidence_high_path)
    if not os.path.exists(result_confidence_low_path):
        os.mkdir(result_confidence_low_path)
    for image in sorted(os.listdir(srcdata_path)):
        srcdata_image_path = os.path.join(srcdata_path, image)
        srcdata_image = cv2.imread(srcdata_image_path, 0)
        resdata_high_imgae = np.zeros(shape=srcdata_image.shape)
        resdata_low_imgae = np.zeros(shape=srcdata_image.shape)
        for i in range(srcdata_image.shape[0]):
            for j in range(srcdata_image.shape[1]):
                if srcdata_image[i][j] > thrs_high:
                    resdata_high_imgae[i][j] = 255
                elif srcdata_image[i][j] > thrs_low:
                    resdata_low_imgae[i][j] = 255

        cv2.imwrite(os.path.join(result_confidence_high_path, image), resdata_high_imgae)
        cv2.imwrite(os.path.join(result_confidence_low_path, image), resdata_low_imgae)

# division_confidence_by_threshold(thrs_high=192, thrs_low=64)


def cal_threshold(image, lamda=0.1):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])  # show the data distrbution using histogram
    amount = 0
    for t in range(256):
        amount += hist[t]
    sum = 0
    thrs = 200
    for t in range(256):
        sum = sum + hist[t]
        if sum >= amount * (1 - lamda):
            thrs = t
            break
    # print('Thr:{}'.format(thrs))
    return thrs

def division_confidence_by_proportion():
    srcdata_path = './datasets/crack500/save/mask_CAM'
    result_path = './datasets/crack500/train/mask_CAM_proportion/c=0.05/mask_CAM'
    result_confidence_high_path = './datasets/crack500/train/mask_CAM_proportion/c=0.05/mask_CAM_proportion_confidence_high'
    result_confidence_low_path = './datasets/crack500/train/mask_CAM_proportion/c=0.05/mask_CAM_proportion_confidence_low'
    if not os.path.exists(result_confidence_high_path):
        os.makedirs(result_confidence_high_path)
    if not os.path.exists(result_confidence_low_path):
        os.makedirs(result_confidence_low_path)
    for image in sorted(os.listdir(srcdata_path)):
        srcdata_image_path = os.path.join(srcdata_path, image)
        srcdata_image = cv2.imread(srcdata_image_path, 0)

        _, pre_fused = cv2.threshold(srcdata_image, thresh=cal_threshold(srcdata_image, 0.15), maxval=255, type=cv2.THRESH_BINARY)

        # dividing merge_cam into confidence_high and confidence_low
        _, resdata_high_imgae = cv2.threshold(srcdata_image, thresh=cal_threshold(srcdata_image, 0.05), maxval=255,
                                                     type=cv2.THRESH_BINARY)
        resdata_low_imgae = pre_fused - resdata_high_imgae

        cv2.imwrite(os.path.join(result_path, image), srcdata_image)
        cv2.imwrite(os.path.join(result_confidence_high_path, image), resdata_high_imgae)
        cv2.imwrite(os.path.join(result_confidence_low_path, image), resdata_low_imgae)

#division_confidence_by_proportion()