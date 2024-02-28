###########################################################################################################################
#################################### reduce data from BGR image to YUV422(UYVY format) ###################################
###########################################################################################################################
# Default imports
import cv2
import numpy as np
import time
# Prepare BGR input (OpenCV uses BGR color ordering and not RGB):
img = cv2.imread('cat.bmp')

start = time.time()
# Split channles, and convert to float
b = img[:,:,0].astype(float)
g = img[:,:,1].astype(float)
r = img[:,:,2].astype(float)

rows, cols = r.shape

# Compute Y, U, V according to the formula described here:
# https://developer.apple.com/documentation/accelerate/conversion/understanding_ypcbcr_image_formats
# U applies Cb, and V applies Cr

# Use BT.709 standard "full range" conversion formula
y = 0.2126*r + 0.7152*g + 0.0722*b
u = 0.5389*(b-y) + 128
v = 0.6350*(r-y) + 128

# Downsample u horizontally
u = cv2.resize(u, (cols//2, rows))      #, interpolation=cv2.INTER_LINEAR

# Downsample v horizontally
v = cv2.resize(v, (cols//2, rows))      #, interpolation=cv2.INTER_LINEAR

# Convert y to uint8 with rounding:
y = np.round(y).astype(np.uint8)

# Convert u and v to uint8 with clipping and rounding:
u = np.round(np.clip(u, 0, 255)).astype(np.uint8)
v = np.round(np.clip(v, 0, 255)).astype(np.uint8)


# y크기의 uv matrix 생성
uv = np.zeros_like(y)

# YUVformat : YUV422 packed format
uv[:, 0::2] = u
uv[:, 1::2] = v

# Merge y and uv channels
uyvy = cv2.merge((uv, y))

print("time :", time.time() - start)

# 파일 경로 및 이름 정의
uyvy_file_path = '/colorextraction/workspace/Data_Reduction/result/uyvy_reduction.yuv'

# .yuv 파일로 저장
with open(uyvy_file_path, 'wb') as yuv_file:
    yuv_file.write(uyvy.tobytes())


# ##########검사
# # Convert yuv422 to BGR for display and saving
# bgr_output = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)

# # Save BGR image to PNG
# cv2.imwrite('uyvy_output.png', bgr_output)
# cv2.imshow('uyvy_output', bgr_output)
# cv2.waitKey(0)

# ##uyvy와 img에서  한 pixel의 bit 수 더한 값 출력
# print(img.dtype)
# print("img size : ", img.size)

# print(uyvy.dtype)
# print("uyvy size : ", uyvy.size)

# img_bit = img.itemsize*8
# uyvy_bit = uyvy.itemsize*8

# print("data size_img (size * bit) : ", img_bit * img.size)
# print("data size_uyvy (size * bit : ", uyvy_bit * uyvy.size)

