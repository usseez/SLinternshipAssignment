
import cv2
import sys
import numpy as np
import matplotlib.pylab as plt

def imshow(title, image) :
    plt.title(title)
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else :
        image[:,:,1] = image[:,:,0]
        image[:,:,2] = image[:,:,0]

def CtoG(image):
    image[:,:,1] = image[:,:,0]
    image[:,:,2] = image[:,:,0]

#img 불러오기
image = cv2.imread('building.png')
image_color = cv2.imread('building.png', cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray_r = cv2.imread('building.png', cv2.IMREAD_GRAYSCALE)
image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
height, width, channel = image.shape #불러온 이미지 크기 구하기


#히스토그램 분석_흑백
histsize = [256]
histrange = [0, 256]

#히스토그램 분석_color
bgr_planes = cv2.split(image)
colors = ['b', 'g', 'r']
for (p, c) in zip(bgr_planes, colors):
    hist = cv2.calcHist([p], [0], None, histsize, histrange)
    plt.plot(hist, color=c)
    
plt.show()   


dst1 = cv2.subtract(image_color, image_gray)
cv2.imshow("dst1", dst1)
#bgr channel separation
##RGB 이미지의 각 색상 채널을 분리하여 휘도와 색차 정보를 얻을 수 있습니다. 
# 예를 들어, 이미지를 각각의 R, G, B 채널로 분리하고 이 중 하나를 휘도로 사용하며, 나머지 두 개를 색차로 사용할 수 있습니다.##
b,g,r = cv2.split(image)


image_gray = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)
plt.subplot(1,2,1)
cv2.imshow('image_gray', image_gray)
cv2.waitKey(0)

hist = cv2.calcHist([image_gray], [0], None, histsize, histrange)
plt.plot(hist)
plt.show()