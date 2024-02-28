import cv2
import sys
import numpy as np
import matplotlib.pylab as plt
import time



#img 불러오기
image = cv2.imread('cat.bmp')



# image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
height, width, channel = image.shape #불러온 이미지 크기 구하기

# 불러왔는지 확인
if image is None:
    print('Image load failed!')
    sys.exit()


#def RGB2HSV(RGB):
    # # HSV 색상을 얻기 위해서는 array 타입이 float이 되어야 계산할 수 있다
    # # RGB_array = np.array(RGB).astype(np.float64)
    
    # # 변환할 HSV 이미지 생성
    # HSV = np.array(RGB)     #.astype(np.float64)

    

    # # 이미지 크기만큼 for 루프
    # for i in range(height):
    #     for j in range(width):
    #         # 공식 따라서 구현
    #         var_R = RGB[i, j, 2] / 255.0
    #         var_G = RGB[i, j, 1] / 255.0
    #         var_B = RGB[i, j, 0] / 255.0

    #         C_Min = min(var_R, var_G, var_B)
    #         C_Max = max(var_R, var_G, var_B)

    #         change = C_Max - C_Min
    #         V = C_Max * 255
            
    #         if C_Max == 0:
    #             S = 0
    #         else:
    #             S = change / C_Max * 255
                
    #         if change == 0:
    #             H = 0
    #         else:
    #             if var_R == C_Max:
    #                 H = (60 * (((var_R - var_B)/change)%6))
    #             elif var_G == C_Max:
    #                 H = (60 * (((var_B - var_R)/change)+2))
    #             elif var_B == C_Max:
    #                 H = (60 * (((var_R - var_B)/change)+4))
                    
    #         HSV[i, j, 0] = H
    #         HSV[i, j, 1] = S
    #         HSV[i, j, 2] = V
    # HSV = HSV.astype(np.uint8)
    # # hsv_image = merge_channels(H, S, V)     
    # return HSV

def RGB2HSV(bgr):
    hue = np.zeros([height, width])
    saturation = np.zeros((height, width))
    value = np.zeros_like((height, width))  

    r, g, b = bgr[:,:,2]/255.0, bgr[:,:,1]/255.0, bgr[:,:,0]/255.0

    cmax = np.maximum(np.maximum(r,g), b)
    cmin = np.minimum(np.minimum(r,g), b)
    delta = cmax - cmin
      

    #calculate val,hue, sat, val>>어케 줄이지
    value = cmax * 255   
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
            # if cmax[i,j] != 0:
            #     saturation[i,j] = delta[i,j] / cmax[i,j] * 255
    saturation = np.where(cmax!=0, (delta / cmax) * 255, 0)
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
#     merged_image = []
#     for r, g, b in zip(red, green, blue):
#         merged_row = []
#         for r_pixel, g_pixel, b_pixel in zip(r, g, b):
#             pixel = [r_pixel, g_pixel, b_pixel]
#             merged_row.append(pixel)
#         merged_image.append(merged_row)
#     return np.array(merged_image)    

def Mask(HSV, color):
    # 범위값과 비교할 hsv 이미지 생성, 파라미터에 있는 HSV를 그냥 쓰면 원소값이 float이 아닌 int로 나옴
    # hsv = np.array(HSV).astype(np.float64)

    # 모든 값은 원소 값이 0 인 마스크 행렬 생성
    mask = np.zeros((height, width))
    # hsv 값과 범위 비교
    lower_condition = (HSV[:,:,0] > lower[color][0]) & (HSV[:,:,1] > lower[color][1]) & (HSV[:,:,2] > lower[color][2]) 
    upper_condition = (HSV[:,:,0] < upper[color][0]) & (HSV[:,:,1] < upper[color][1]) & (HSV[:,:,2] < upper[color][2]) 
    condition = lower_condition & upper_condition
    mask[condition] = 1
    # for i in range(height):
    #     for j in range(width):
    #         # H, S, V 값이 원하는 범위 안(color별 h, s, v 범위)에 들어갈 경우 mask 원소 값을 1로 만든다
    #         if (HSV[i, j, 0] > lower[color][0] and 
    #             HSV[i, j, 1] > lower[color][1] and 
    #             HSV[i, j, 2] > lower[color][2] and
                 
    #             HSV[i, j, 0] < upper[color][0] and 
    #             HSV[i, j, 1] < upper[color][1] and 
    #             HSV[i, j, 2] < upper[color][2]):
                
    #             mask[i, j] = 1
    ##not working
    
    

    # mask[HSV[:,:,0] > lower[color][0] and
    #      HSV[:,:,1] > lower[color][1] and
    #      HSV[:,:,2] > lower[color][2] and
         
    #      HSV[:,:,0] < upper[color][0] and
    #      HSV[:,:,1] < upper[color][1] and
    #      HSV[:,:,2] < upper[color][2]] = 1            
    return mask

def Extraction(image, mask):
    # Object를 추출할 이미지를 생성
    result_img = np.array(image)
    result_img[mask == 0] = 0


    # for 루프를 돌면서 mask 원소 값이 0인 인덱스는 원본 이미지도 0으로 만들어 준다.
    # for i in range(width):
    #     for j in range(height):
    #         if(mask[i, j] == 0):
    #             result_img[i, j, 0] = 0
    #             result_img[i, j, 1] = 0
    #             result_img[i, j, 2] = 0
                
    
                
    return result_img


if __name__ == '__main__':
    # 마스크 색상 범위에 사용할 딕셔너리 정의
    upper = {}
    lower = {}

    upper['red'] = [25, 255, 255]       #Hue range : 0~179, Saturation : 0~255, Value : 0~255
    upper['blue'] = [130, 255, 255]
    upper['green'] = [70, 255, 255]

    lower['red'] = [0, 40, 40]
    lower['blue'] = [110, 40, 40]
    lower['green'] = [40, 40, 40]


    # 추출하고 싶은 색상 입력
    input_color = input("추출하고 싶은 색상을 입력하세요 (red, blue, green) : ")
    
    
    ## 처리속도 측정
    start = time.time()
    # RGB to HSV 변환
    HSV = RGB2HSV(image)
    # cv2.imshow('HSV_golden', HSV_golden)  #BGR순서로 출력됨
    # cv2.imshow('HSV', HSV)  #BGR순서로 출력됨
    # cv2.waitKey()
    # print(HSV)
    # HSV 이미지를 가지고 마스크 생성
    mask = Mask(HSV, input_color)
    # mask를 가지고 원본이미지를 Object 추출 이미지로 변환
    result_image = Extraction(image, mask)

    #mping.imsave("result.jpg", result_image)
    
    
    print("Extraction Working time :", time.time() - start)
    print(result_image)

    # 이미지 보여주기
    cv2.imshow('result_image', result_image)
    cv2.waitKey()
    plt.show()