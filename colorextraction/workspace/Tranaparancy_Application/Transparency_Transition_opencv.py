## https://minimin2.tistory.com/131참고
import cv2
import numpy as np
import matplotlib.pylab as plt


win_name = 'Alpha blending'     # 창 이름
trackbar_name = 'fade'          # 트렉바 이름

# ---① 트렉바 이벤트 핸들러 함수
def onChange(x):
    alpha = x/100
    dst = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0) 
    cv2.imshow(win_name, dst)


# Prepare BGR input (OpenCV uses BGR color ordering and convert image to BGRA):
img1 = cv2.imread('cat.bmp')
img2 = cv2.imread('cloud.bmp')
rgba_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGBA)
rgba_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGBA)
print(img1.shape)
print(img2.shape)

# print(rgba_img)

#split each chroma data
r_data = rgba_img1[:,:,0]
g_data = rgba_img1[:,:,1]
b_data = rgba_img1[:,:,2]
alpha_data = rgba_img1[:,:,3]


# alpha bleding
alpha = 0.5
blended = img1 * alpha + img2 * (1-alpha)
blended = blended.astype(np.uint8)

blended2 = cv2.addWeighted(img1, alpha, img2, (1-alpha), 0) 

blended = cv2.cvtColor(blended, cv2.COLOR_BGRA2RGBA)
blended2 = cv2.cvtColor(blended2, cv2.COLOR_BGRA2RGBA)
blended3 = cv2.add(img1, img2)


## plot images
plt.figure(figsize=(10,  10))
plt.subplot(2,2,1)
plt.imshow(rgba_img1)
plt.title('rgba_img')
plt.subplot(2,2,2)
plt.imshow(rgba_img2)
plt.title('rgba_back')

plt.subplot(2,2,3)
plt.imshow(blended)
plt.title('blended_using Numpy directly')
plt.subplot(2,2,4)
plt.imshow(blended2)
plt.title('blended_using_addWeighted')

# ---③ 이미지 표시 및 트렉바 붙이기
cv2.imshow(win_name, img1)
cv2.createTrackbar(trackbar_name, win_name, 0, 100, onChange)

cv2.waitKey()
cv2.destroyAllWindows()

plt.show()