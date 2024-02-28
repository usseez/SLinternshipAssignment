# Default imports
import sys
import cv2
import time

#이미지 로딩
image = cv2.imread("cat.bmp")

# 불러왔는지 확인
if image is None:
    print('Image load failed!')
    sys.exit()

image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
image_i420 = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
image_iyuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_IYUV)
image_yv12 = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_YV12)

# cv2.imwrite('/colorextraction/workspace/Data_Reduction/result/image_yuv.bin', image_yuv)
# cv2.imwrite('/colorextraction/workspace/Data_Reduction/result/image_i420.bin', image_i420)
# cv2.imwrite('/colorextraction/workspace/Data_Reduction/result/image_iyuv.bin', image_iyuv)
# cv2.imwrite('/colorextraction/workspace/Data_Reduction/result/image_yv12.bin', image_yv12)

# 파일 경로 및 이름 정의
image_yuv_file_path = '/colorextraction/workspace/Data_Reduction/result/image_yuv.yuv'
image_i420_file_path = '/colorextraction/workspace/Data_Reduction/result/image_i420.yuv'
image_iyuv_file_path = '/colorextraction/workspace/Data_Reduction/result/image_iyuv.yuv'
image_yv12_file_path = '/colorextraction/workspace/Data_Reduction/result/image_yv12.yuv'


with open(image_yuv_file_path, 'wb') as yuv_file:
    yuv_file.write(image_yuv.tobytes())
with open(image_i420_file_path, 'wb') as i420_file:
    i420_file.write(image_i420.tobytes())
with open(image_iyuv_file_path, 'wb') as iyuv_file:
    iyuv_file.write(image_iyuv.tobytes())
with open(image_yv12_file_path, 'wb') as yv12_file:
    yv12_file.write(image_yv12.tobytes())
        
    
    