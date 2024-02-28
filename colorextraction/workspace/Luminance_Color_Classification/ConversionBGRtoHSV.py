import cv2
import numpy as np
import time
import matplotlib.pylab as plt
import matplotlib.image as mping


HUE_DEGREE = 512
# BGR 이미지 불러오기
bgr_image = cv2.imread('cat.bmp')

height, width, channel = bgr_image.shape


np.seterr(divide='ignore', invalid='ignore')

## 처리속도 측정
start = time.time()

# https://www.geeksforgeeks.org/program-change-rgb-color-model-hsv-color-model/
def bgr2hsv(bgr):
    r = bgr[:,:,2]/255.0
    g = bgr[:,:,1]/255.0
    b = bgr[:,:,0]/255.0

    cmax = np.maximum(np.maximum(r,g), b)
    cmin = np.minimum(np.minimum(r,g), b)
    
    delta = cmax - cmin
    
    hue = np.zeros([height, width])
    saturation = np.zeros((height, width))
    value = np.zeros_like((height, width))
    
    print("Working time :", time.time() - start)
    
    
    #calculate val,hue, sat, val>>어케 줄이지
          
    # for i in range(height):
    #     for j in range(width):
    #         # if delta[i,j] == 0 :
    #         #     hue[i,j] = 0
    #         if cmax[i,j] == r[i,j]:
    #             hue[i,j] = (((g[i,j]-b[i,j]) /delta[i,j]) % 6) * 30
    #         elif cmax[i,j] == g[i,j]:
    #             hue[i,j] = (((b[i,j]-r[i,j]) /delta[i,j]) + 2) * 30
    #         else :
    #             hue[i,j] = (((r[i,j]-g[i,j]) /delta[i,j]) + 4) * 30
    r_eq_cmax = cmax == r
    g_eq_cmax = cmax == g
    b_eq_cmax = cmax == b
    
    hue[r_eq_cmax] = (((g[r_eq_cmax]-b[r_eq_cmax]) /delta[r_eq_cmax]) % 6) * 30
    hue[g_eq_cmax] = (((b[g_eq_cmax]-r[g_eq_cmax]) /delta[g_eq_cmax]) + 2) * 30 
    hue[b_eq_cmax] = (((r[b_eq_cmax]-g[b_eq_cmax]) /delta[b_eq_cmax]) + 4) * 30 
    
    saturation = np.where(cmax!=0, (delta / cmax) * 255, 0)
    value = cmax * 255 
            # if cmax[i,j] != 0:
            #     saturation[i,j] = delta[i,j] / cmax[i,j] * 255
    
    ##not working :  NumPy boolean array indexing assignment requires a 0 or 1-dimensional input, input has 2 dimensions
    # saturation[cmax != 0] = delta / cmax * 255: 오른쪽 항 = 1이면 경고 안뜸 : 1차원이어야 하나......   

    # print(hue_r.shape)        #1280*800
    # print((cmax == r).shape)  #1280*800
    # print((hue).shape)        #1280*800
    
    # hue[cmax == r] = hue_r
    # # hue[cmax == r] = (((g-b) /delta) % 6) * 30  
    # hue[cmax == g] = (((b-r) /delta) + 2) * 30 
    # hue[cmax == b] = (((r-g) /delta) + 4) * 30
            
    
    
    
    # hsv_image = merge_channels(hue, saturation, value)
    hsv_image = np.zeros((height, width, channel))
    hsv_image[:, :, 0] = hue
    hsv_image[:, :, 1] = saturation
    hsv_image[:, :, 2] = value
    hsv_image = hsv_image.astype(np.uint8)
    # hue = hue.astype(np.uint8)
    # saturation = saturation.astype(np.uint8)
    # value = value.astype(np.uint8)
    
    return hsv_image
# def merge_channels(red, green, blue):
    merged_image = []
    for r, g, b in zip(red, green, blue):
        merged_row = []
        for r_pixel, g_pixel, b_pixel in zip(r, g, b):
            pixel = [r_pixel, g_pixel, b_pixel]
            merged_row.append(pixel)
        merged_image.append(merged_row)
    return np.array(merged_image)    




hsv_image = bgr2hsv(bgr_image)

print("Working time :", time.time() - start)






# YUV split
h = hsv_image[:,:,0]
s = hsv_image[:,:,1]
v = hsv_image[:,:,2]

restored_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2RGB)


# show images : y,u,v, yuv, 
plt.figure(figsize=(20, 10))
plt.subplot(2,3,1)
plt.imshow(h, cmap='gray')
plt.title('h')
plt.subplot(2,3,2)
plt.imshow(s, cmap='gray')
plt.title('s')
plt.subplot(2,3,3)
plt.imshow(v, cmap='gray')
plt.title('v')
plt.subplot(2,3,4)
plt.imshow(hsv_image)
plt.title('hsv_image')
plt.subplot(2,3,5)
plt.imshow(bgr_image)
plt.title('bgr_image')
plt.subplot(2,3,6)
plt.imshow(restored_image)
plt.title('restored_image')

plt.show()





##파일 저장
hsv_image_file_path = 'D:/colorextraction/workspace/Luminance_Color_Classification/result/hsv_image.bmp'
# goldenfile.bin 파일로 저장
with open(hsv_image_file_path, 'wb') as hsv_file:
    hsv_file.write(hsv_image.tobytes())
    








# def Mask(HSV, color):
#     # 범위값과 비교할 hsv 이미지 생성, 파라미터에 있는 HSV를 그냥 쓰면 원소값이 float이 아닌 int로 나옴
#     hsv = np.array(HSV).astype(np.float64)

#     # HSV 이미지의 width, height 저장
#     width, height = HSV.shape[:2]

#     # 모든 값은 원소 값이 0 인 마스크 행렬 생성
#     mask = np.zeros((width, height))

#     # hsv 값과 범위 비교
#     for i in range(width):
#         for j in range(height):
#             # H, S, V 값이 원하는 범위 안에 들어갈 경우 mask 원소 값을 1로 만든다
#             if hsv[i, j, 0] > lower[color][0] and hsv[i, j, 1] > lower[color][1] and hsv[i, j, 2] > lower[color][2] and hsv[i, j, 0] < upper[color][0] and hsv[i, j, 1] < upper[color][1] and hsv[i, j, 2] < upper[color][2]:
#                 mask[i, j] = 1
                
#     return mask

# def Extraction(image, mask):
#     # Object를 추출할 이미지를 생성
#     result_img = np.array(image)

#     # RGB 이미지의 width, height 저장
#     width, height = image.shape[:2]

#     # for 루프를 돌면서 mask 원소 값이 0인 인덱스는 원본 이미지도 0으로 만들어 준다.
#     for i in range(width):
#         for j in range(height):
#             if(mask[i, j] == 0):
#                 result_img[i, j, 0] = 0
#                 result_img[i, j, 1] = 0
#                 result_img[i, j, 2] = 0
                
#     return result_img


# if __name__ == '__main__':
#     # 마스크 색상 범위에 사용할 딕셔너리 정의
#     upper = {}
#     lower = {}

#     upper['orange'] = [100, 1, 1]
#     upper['blue'] = [300, 1, 1]
#     upper['green'] = [180, 0.7, 0.5]

#     lower['orange'] = [0, 0.7, 0.5]
#     lower['blue'] = [70, 0.7, 0.2]
#     lower['green'] = [101, 0.15, 0]

#     # 이미지 파일을 읽어온다
#     input_image = mping.imread('cat.bmp')

#     # 추출하고 싶은 색상 입력
#     input_color = input("추출하고 싶은 색상을 입력하세요 (orange, blue, green) : ")

#     # RGB to HSV 변환
#     HSV = bgr2rgb(input_image)

#     # HSV 이미지를 가지고 마스크 생성
#     mask = Mask(HSV, input_color)

#     # mask를 가지고 원본이미지를 Object 추출 이미지로 변환
#     result_image = Extraction(input_image, mask)

#     #mping.imsave("result.jpg", result_image)

#     # 이미지 보여주기
#     imgplot = plt.imshow(result_image)

#     plt.show()


