# Default imports
import cv2
import numpy as np
import time


# Prepare BGR input (OpenCV uses BGR color ordering and not RGB):
img = cv2.imread('cat.bmp')
height, width = img.shape[:2]

## 처리속도 측정
start = time.time()
def bgr2gray(bgr):
    r, g, b = bgr[:,:,0], bgr[:,:,1], bgr[:,:,2]
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)
    print(gray)
    return gray

# BGR to RGB function
# def bgr2rgb(bgr):
#     rgb = bgr[:, :, ::-1]
#     return rgb
 
# img_rgb = bgr2rgb(img)
img_gray = bgr2gray(img)
print("time :", time.time() - start)

# cv2.imshow('img_gray', img_gray)
# cv2.waitKey()
cv2.imwrite('/colorextraction/workspace/Data_Reduction/graycat.bmp', img_gray)


