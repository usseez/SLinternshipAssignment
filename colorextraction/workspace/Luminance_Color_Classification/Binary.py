###########################################################################################################################
################################################## Color, Luminance Split##################################################
###########################################################################################################################

import cv2
import sys
import numpy as np
import matplotlib.pylab as plt
import time


#img 불러오기
image = cv2.imread('cat.bmp')
height, width, channel = image.shape #불러온 이미지 크기 구하기

# 불러왔는지 확인
if image is None:
    print('Image load failed!')
    sys.exit()
 
 
start = time.time()  
## BGR to RGB function
# def bgr2rgb(bgr):
#     rgb = bgr[:, :, ::-1]
#     return rgb

# def rgb2gray(rgb):
def bgr2gray(bgr):
        
    
    r, g, b = bgr[:,:,2], bgr[:,:,1], bgr[:,:,0]
    gray = np.zeros([height,width])
    # for i in range(height):
    #     for j in range(width):
    #         gray[i,j] = (0.2989 * r[i,j] + 0.5870 * g[i,j] + 0.1140 * b[i,j])
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
    # gray = (0.3 * r + 0.6 * g + 0.1 * b)
    gray = gray.astype(np.uint8)

    return gray
 


##### 각 픽셀값 분류하기
#이진화 : Binariztion
def binarization(image) : #, threshold):
    
    #image copy떠서 함수 안에서 처리
    #안에서 image리스트를 새로 만들어서 하기
    binarized_image = np.zeros([height,width])
    # 시간이 오래 걸린다
    # for i in range(height):
    #     for j in range(width):
    #         if(image[i,j] > threshold):
    #             binarized_image[i,j] = 1
    #         else :
    #             binarized_image[i,j] = 0
    
    #조건 만족하는 index를 뽑아서 위치에 1을 넣어준다
    binarized_image[image > threshold] = 1
    # binarized_image[image <= threshold] = 0
    
    # binarized_image(np.where(image > threshold)) = 1
    # binarized_image(np.where(image <= threshold)) = 0
    return binarized_image

if __name__ == '__main__':
    
    
    #gray이진화(binarization)
    # image_rgb = bgr2rgb(image)
    threshold = 150     #0~255
    # input_thresh = int(input("기준값을 입력하세요 (0~255) : "))
    print("1Working time :", time.time() - start)
    image_gray = bgr2gray(image)
    print("2Working time :", time.time() - start)

    bin_image = binarization(image_gray) #포인터처럼 썼기 때문에 image_gray 자체도 바뀐것임....
    print("3Working time :", time.time() - start)



    cv2.imshow("binary_image", bin_image)
    cv2.waitKey()



    #bin127  = cv2.threshold(image_gray,  127, 255, cv2.THRESH_BINARY)[1]
    # #적응형 이진화
    # mean_bin = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 0)
    # gaus_bin = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 0) ## opencv안쓰고 구현해보기
    # #plot
    # fig, subs = plt.subplots(ncols = 3, figsize = (15, 5), sharex = True, sharey = True)
    # subs[0].set_title('bin127')
    # subs[0].imshow(bin127, cmap = 'gray')

    # subs[1].set_title('ADAPTIVE_THRESH_MEAN_C')
    # subs[1].imshow(mean_bin, cmap = 'gray')

    # subs[2].set_title('ADAPTIVE_THRESH_GAUSSIAN_C')
    # subs[2].imshow(gaus_bin, cmap = 'gray')

    # plt.show()


    # 결과 표시(평가를 어떻게 할 것인가?)



    # # 컬러 영상 속성 확인
    # print('image.shape:', image.shape, '\nimage.dtype:', image.dtype)  # image.shape: (width, height, channel(1280, 800, 3)) # image.dtype: unit8
    # print('image_gray.shape:', image_gray.shape, '\nimage_gray.dtype:', image_gray.dtype)


    # # show image
    # cv2.imshow("Original Image", image)
    # cv2.imshow("Lab Image", image_lab)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

