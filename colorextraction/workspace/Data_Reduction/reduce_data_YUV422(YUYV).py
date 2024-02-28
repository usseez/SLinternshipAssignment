###########################################################################################################################
#################################### reduce data from BGR image to YUV422(YUYV format) ####################################
###########################################################################################################################

# Default imports
import cv2
import numpy as np
import sys
import time
# Prepare BGR input (OpenCV uses BGR color ordering and not RGB):
img = cv2.imread('cat.bmp')

# 불러왔는지 확인
if img is None:
    print('Image load failed!')
    sys.exit()



#bgr to yuyv함수
def bgr2yuyv(bgr):
    
    b = bgr[:,:,0].astype(float)
    g = bgr[:,:,1].astype(float)
    r = bgr[:,:,2].astype(float)

    rows, cols = r.shape

    # Use BT.709 standard "full range" conversion formula
    y = 0.2126*r + 0.7152*g + 0.0722*b
    u = 0.5389*(b-y) + 128
    v = 0.6350*(r-y) + 128

    # Downsample u,v horizontally
    u = cv2.resize(u, (cols//2, rows))
    v = cv2.resize(v, (cols//2, rows))

    # Convert y to uint8 with rounding:
    y = y.astype(np.uint8)
    #y = np.round(y).astype(np.uint8)
    print(u,v)
    # Convert u and v to uint8 with clipping and rounding: clipping - 0~255값으로 변환
    u = np.round(np.clip(u, 0, 255)).astype(np.uint8)
    v = np.round(np.clip(v, 0, 255)).astype(np.uint8)


    # y크기의 uv matrix 생성
    uv = np.zeros_like(y)

    # YUVformat : YUV422 packed format
    uv[:, 0::2] = u
    uv[:, 1::2] = v

    # Merge y and uv channels
    yuyv = cv2.merge((y, uv))
    # print(yuyv)
    return yuyv

#시간 측정
start = time.time()

# convert bgr2yuyv
yuyv = bgr2yuyv(img)

print("time :", time.time() - start)

# 파일 경로 및 이름 정의
yuv_file_path = '/colorextraction/workspace/Data_Reduction/result/yuyv_reduction.yuv'

# .yuv 파일로 저장
with open(yuv_file_path, 'wb') as yuv_file:
    yuv_file.write(yuyv.tobytes())
    
    
# #####################
# ##yuyv와 img에서  한 pixel의 bit 수 더한 값 출력
# print(img.dtype)
# print("img size : ", img.size)

# print(yuyv.dtype)
# print("yuyv size : ", yuyv.size)

# img_bit = img.itemsize*8
# yuyv_bit = yuyv.itemsize*8

# print("data size_img (size * bit) : ", img_bit * img.size)
# print("data size_yuyv (size * bit : ", yuyv_bit * yuyv.size)

# # Convert yuv422 to BGR for display and saving(검증)
# bgr_output = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)

# # Save BGR image to PNG
# cv2.imwrite('yuyv_output.png', bgr_output)
# cv2.imshow('yuyv_output', bgr_output)
# cv2.waitKey(0)