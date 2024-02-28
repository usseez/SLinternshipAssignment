import cv2
import sys
import numpy as np
import matplotlib.pylab as plt

bgr_image = cv2.imread('cat.bmp')

lab_plt = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab)
hsv_plt = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
restore_lab_plt = cv2.cvtColor(lab_plt, cv2.COLOR_Lab2RGB)

l = lab_plt[:,:,0]
a = lab_plt[:,:,1]
b = lab_plt[:,:,2]

print(lab_plt)

plt.figure(figsize=(20, 10))
plt.subplot(2,3,1)
plt.imshow(lab_plt)
plt.title('lab_plt')
plt.subplot(2,3,2)
plt.imshow(restore_lab_plt)
plt.title('restore_lab_plt')
plt.subplot(2,3,4)
plt.imshow(l, cmap='gray')
plt.title('l')
plt.subplot(2,3,5)
plt.imshow(a, cmap='gray')
plt.title('a')
plt.subplot(2,3,6)
plt.imshow(b, cmap='gray')
plt.title('b')



plt.show()