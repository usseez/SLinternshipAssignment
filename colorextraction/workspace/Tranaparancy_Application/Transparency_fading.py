## from https://stackoverflow.com/questions/69579255/how-to-fade-image-in-python
import matplotlib.pylab as plt
import matplotlib.image as img
from PIL import Image
import numpy as np
import cv2
import time


def bgr2rgb(bgr):
    rgb = bgr[:, :, ::-1]
    return rgb

#rgb2rgba function
def rgb2rgba(rgb, value):
    height, width, _ = rgb.shape
    rgba_image = np.zeros((height, width, 4))

    rgba_image[:,:,0:3] = rgb
    rgba_image[:,:,3] = value
    # for y in range(width):
    #     for x in range(height):
    #         rgba_image[x,y,3]= value
    rgba_image = rgba_image.astype(np.uint8)
    return rgba_image




#수직방향 fade out 함수
#def fade_image_vertical(image_v, p1, p2, flow_up=False):
    height, width, _ = image_v.shape    #이미지 세로*가로 크기 저장
    fade_range = list(range(int(height * p1), int(height * p2)))    #fade할 범위 2개의 list로 지정, 
##    fade_range = fade_range[::-1] if flow_up else fade_range        #??왜있나??flowup이면(위로갈수록 fade_out) 리스트 역으로:...5,4,3,2,1,0//아니면 그대로
    
    for y in fade_range:
        if flow_up:                                                     #위로 갈수록 fade out
            alpha = int((y - height * p1) / height / (p2 - p1) * 255)   # y열 숫자가 클수록 alpha 증가, 투명도 감소
        else:                                                                   #아래로 갈수록 fade out
            alpha = 255 - int((y - height * p1) / height / (p2 - p1) * 255)     #y열 숫자 클수록 alpha 감소, 투명도 증가
        for x in range(width):                                            # width에 대하여 반복한다
            image_v[y, x, 3] = alpha                                      #image_v의 3번째 채널(?)에 앞에서 계산했던 alpha값을 계속 대입해준다
    return image_v

#def fade_image_horizontal(image_h, p1, p2, flow_left=False):
    
    height, width, _ = image_h.shape
    fade_range = list(range(int(width*p1), int(width*p2)))
    fade_range = fade_range[::-1] if flow_left else fade_range

    for x in fade_range:
        if flow_left:
            alpha = int((x - width*p1) / width / (p2-p1) * 255)
        else:
            alpha = 255-int((x - width*p1) / width / (p2-p1) * 255)   
        for y in range(height):
            image_h[y, x, 3] = alpha        
    return image_h


def fade_image_edge(image_e, fade_distance):
    # 이미지의 크기 가져오기
    height, width, _ = image_e.shape
    # 가장자리에서 투명도 적용
    for y in range(height):
        for x in range(width):                      #0행0열부터 0행1열, 0행 2열... 순으로 적용
            alpha = min(255, int(min(x, y, width - x, height - y)) / fade_distance * 255)   #alpha값은 행과 열이 이미지 크기의 극단에 있을수록 작아짐(투명도 증가)
            image_e[y, x, 3] = alpha
   
    return image_e

#import image
image_original = cv2.imread('cat.bmp')
## 처리속도 측정
start = time.time()
image_rgb = bgr2rgb(image_original)

image_rgba = rgb2rgba(image_rgb, 175)
# image_v = image_rgba.copy()
# image_h = image_rgba.copy()
image_e = image_rgba.copy()


# #fade out each direction and Save it
# fade_image_vertical(image_v, 0.1, 1.0 , flow_up=False)    #bottom, 아래로 갈수록 fade out
# fade_image_vertical(image_v, 0  , 0.1, flow_up=True)      #top, 위로갈수록 fade out
# cv2.imwrite('fade_image_vertical.png', image_v)

# fade_image_horizontal(image_h, 0.75, 1.0 , flow_left=False)   #right, right로 갈수록 fade out
# fade_image_horizontal(image_h, 0  , 0.25, flow_left=True)     #left, left로 갈수록 fade out
# cv2.imwrite('fade_image_horizontal.png', image_h)

fade_distance = 510
fade_image_edge(image_e, fade_distance)
print("Working time :", time.time() - start)
# cv2.imwrite('fade_image_edge.png', image_e)

#Show results
plt.figure(figsize=(10,  10))
plt.subplot(1,1,1)
plt.axis('off')  # Turn off axis labels
plt.imshow(image_e)
plt.title('image_e')
plt.show()


# faded_v = img.imread('fade_image_vertical.png')
# faded_h = img.imread('fade_image_horizontal.png')
# fade_edge = img.imread('fade_image_edge.png')

# plt.figure(figsize=(10,  10))
# plt.subplot(2,2,1)
# plt.axis('off')  # Turn off axis labels
# plt.imshow(image_rgb)
# plt.title('image_original')
# plt.subplot(2,2,2)
# plt.axis('off')  # Turn off axis labels
# plt.imshow(faded_v)
# plt.title('faded_v')
# plt.subplot(2,2,3)
# plt.axis('off')
# plt.imshow(faded_h)
# plt.title('faded_h')
# plt.subplot(2,2,4)
# plt.axis('off')
# plt.imshow(fade_edge)
# plt.title('fade_image_edge')

