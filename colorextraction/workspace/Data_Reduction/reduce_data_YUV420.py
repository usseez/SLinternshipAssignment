###########################################################################################################################
#################################### reduce data from BGR image to YUV420(NV12 format) ####################################
###########################################################################################################################
# Default imports
import cv2
import numpy as np
import time

# Prepare BGR input (OpenCV uses BGR color ordering and not RGB):
img = cv2.imread('cat.bmp')

#시작시간 저장
start = time.time()
def rgb2nv12(rgb):
    
    
    # Split channles, and convert to float
    b = rgb[:,:,0].astype(float)
    g = rgb[:,:,1].astype(float)
    r = rgb[:,:,2].astype(float)
    height, width = img.shape[:2]
    rows, cols = r.shape

    # Use BT.709 standard "full range" conversion formula
    y = 0.2126*r + 0.7152*g + 0.0722*b
    u = 0.5389*(b-y) + 128
    v = 0.6350*(r-y) + 128

    # Downsample  U,V channels by a factor of x2 in each axis
    u = cv2.resize(u, (cols//2, rows//2))
    v = cv2.resize(v, (cols//2, rows//2))

    # Convert y to uint8 with rounding:
    y = np.round(y).astype(np.uint8)

    # Convert u and v to uint8 with clipping and rounding:
    u = np.round(np.clip(u, 0, 255)).astype(np.uint8)
    v = np.round(np.clip(v, 0, 255)).astype(np.uint8)

    ## Chroma elements interleaving - arrange U,V elements as U,V,U,V...
    # y크기의 uv matrix 생성
    uv = np.zeros([height//2, width])
    # YUVformat : YUV420 planar format
    uv[:, 0::2] = u #모든 행에서 짝수열 선택
    uv[:, 1::2] = v

    # Merge y and uv channels
    yuv420 = (np.vstack((y, uv)).astype(np.uint8))
    
    return yuv420

yuv420 = rgb2nv12(img)
print("time :", time.time() - start)

# 파일 경로 및 이름 정의
yuv_file_path = '/colorextraction/workspace/Data_Reduction/result/nv12_reduction.yuv'
# .yuv 파일로 저장
with open(yuv_file_path, 'wb') as yuv_file:
    yuv_file.write(yuv420.tobytes())
    
    
    
    
    
    
    
    
    
# print("img.shape : ", img.shape)
# print("yuv420.shape : ", yuv420.shape)


# ##yuyv와 img에서  한 pixel의 bit 수 더한 값 출력
# print("img.dtype : ", img.dtype)
# print("img size : ", img.size)

# print("yuv420.dtype : ", yuv420.dtype)
# print("yuv420 size : ", yuv420.size)

# #byte size(.itemsize) * 8 = bit 크기 구하기
# img_bit = img.itemsize*8
# yuv420_bit = yuv420.itemsize*8

# print("img size * bit : ", img_bit * img.size)
# print("yuv420 size * bit : ", yuv420_bit * yuv420.size)



