import cv2
import numpy as np
import matplotlib.pylab as plt
import time

# Prepare BGR input (OpenCV uses BGR color ordering and not RGB):
img = cv2.imread('cat.bmp')

height, width, channel = img.shape
## 처리속도 측정
start = time.time()
def bgr2rgb(bgr):
    rgb = bgr[:, :, ::-1]
    return rgb

#rgb2rgba function
def rgb2rgba(rgb, value):
    height, width, _ = img.shape
    rgba_image = np.zeros((height, width, 4))

    rgba_image[:,:,0:3] = rgb
    rgba_image[:,:,3] = value
    
    # for y in range(width):
    #     for x in range(height):
    #         rgba_image[x,y,3]= value
    rgba_image = rgba_image.astype(np.uint8)
    return rgba_image

#Convert rgb2rgba
rgb = bgr2rgb(img)
value = 100
rgba = rgb2rgba(rgb, value)
print("Working time :", time.time() - start)
#plot image
plt.figure(figsize=(10,  10))
plt.subplot(1,2,1)
plt.axis('off')  # Turn off axis labels
plt.imshow(rgb)
plt.title('rgb')
plt.subplot(1,2,2)
plt.axis('off')  # Turn off axis labels
plt.imshow(rgba)
plt.title('rgba')

plt.show()