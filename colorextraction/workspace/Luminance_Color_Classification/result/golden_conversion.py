import cv2
import sys
import numpy as np
import matplotlib.pylab as plt

bgr_image = cv2.imread('cat.bmp')
hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
yuv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YUV)
cv2.imshow('hsv', hsv)
cv2.imshow('yuv', yuv)
cv2.waitKey()