# Default imports
import sys
import numpy as np
import cv2
import time


#이미지 로딩
image = cv2.imread("cat.bmp")

# 불러왔는지 확인
if image is None:
    print('Image load failed!')
    sys.exit()

## 처리속도 측정
start = time.time()

# BGR to RGB function
# def bgr2rgb(bgr):
#     rgb = bgr[:, :, ::-1]
#     return rgb

#RGB888 to RGB 565 function
def bgr2rgb565(bgr):

    r = (bgr[:,:,2])    #extract red component
    g = (bgr[:,:,1])    #extract green component
    b = (bgr[:,:,0])    #extract blue component
    
    r5 = (r >> 3) << 3      #30->3  0001_1110 -> 0000_0011
    g6 = (g >> 2)           #1001_1110 -> 0010_0111
    b5 = (b >> 3)           #0001_1110 -> 0000_0011
    
    g_upper = g6 >> 3                   #0010_0111 -> 0000_0100
    g_lower = (g6 & 0b0000_0111) << 5      #(0010_0111 -> 0000_0001) -> 0010_0000
    
    upper_comp = g_lower | b5
    lower_comp = r5 | g_upper
    
    rgb565 = np.dstack([upper_comp, lower_comp])
    
    return rgb565


#rgb_image = bgr2rgb(image)
rgb565_image = bgr2rgb565(image)

print("Working time :", time.time() - start)
##파일 저장
rgb565_image_file_path = '/colorextraction/workspace/Data_Reduction/result/rgb565_image.bin'
# goldenfile.bin 파일로 저장
with open(rgb565_image_file_path, 'wb') as bgr_file:
    bgr_file.write(rgb565_image.tobytes())
