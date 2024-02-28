import cv2
import numpy as np
import time
import matplotlib.pylab as plt



# BGR 이미지 불러오기
image = cv2.imread('cat.bmp')
height, width, channel = image.shape
## 처리속도 측정


def bgr_to_rgb(bgr):
    return bgr[:, :, ::-1]

def bgr_to_yuv(bgr):
    # r = bgr[:,:,2]
    # g = bgr[:,:,1]
    # b = bgr[:,:,0]
    # print("Working time :", time.time() - start)
    # y = 0.257 * r + 0.504  * g + 0.098 * b + 16
    # u = -(0.148 * r) - (0.291 * g) + (0.439 * b) + 128
    # v = (0.439 * r) - (0.368 * g) - (0.071 * b) + 128

    y = 0.257 * bgr[:,:,2] + 0.504  * bgr[:,:,1] + 0.098 * bgr[:,:,0] + 16
    u = -(0.148 * bgr[:,:,2]) - (0.291 * bgr[:,:,1]) + (0.439 * bgr[:,:,0]) + 128
    v = (0.439 * bgr[:,:,2]) - (0.368 * bgr[:,:,1]) - (0.071 * bgr[:,:,0]) + 128
    
    # y = y.astype(np.uint8)
    # u = u.astype(np.uint8)
    # v = v.astype(np.uint8)
 

    # yuv_image = merge_channels(y,u,v)
    yuv_image = np.zeros((height, width, channel))


    yuv_image[:,:,0] = y
    yuv_image[:,:,1] = u
    yuv_image[:,:,2] = v


    yuv_image = yuv_image.astype(np.uint8)


    return yuv_image

# def merge_channels(red, green, blue):
#     merged_image = []
#     for r, g, b in zip(red, green, blue):
#         merged_row = []
#         for r_pixel, g_pixel, b_pixel in zip(r, g, b):
#             pixel = [r_pixel, g_pixel, b_pixel]
#             merged_row.append(pixel)
#         merged_image.append(merged_row)
#     return np.array(merged_image)


start = time.time()
# RGB를 YUV로 변환
yuv_image = bgr_to_yuv(image)

print("Working time :", time.time() - start)
# BGR을 RGB로 변환
rgb_image = bgr_to_rgb(image)

# YUV split
y = yuv_image[:,:,0]
u = yuv_image[:,:,1]
v = yuv_image[:,:,2]

restore_bgr = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)

##파일 저장
yuv_image_file_path = 'D:/colorextraction/workspace/Luminance_Color_Classification/result/yuv_image.bin'
# goldenfile.bin 파일로 저장
with open(yuv_image_file_path, 'wb') as yuv_file:
    yuv_file.write(yuv_image.tobytes())
    
    
# show images : y,u,v, yuv, 
plt.figure(figsize=(20, 10))
plt.subplot(2,3,1)
plt.imshow(y, cmap='gray')
plt.title('y')
plt.subplot(2,3,2)
plt.imshow(u, cmap='gray')
plt.title('u')
plt.subplot(2,3,3)
plt.imshow(v, cmap='gray')
plt.title('v')
plt.subplot(2,3,4)
plt.imshow(yuv_image)
plt.title('yuv_image')
plt.subplot(2,3,5)
plt.imshow(rgb_image)
plt.title('rgb_image')
plt.subplot(2,3,6)
plt.imshow(restore_bgr)
plt.title('restore_bgr')

plt.show()