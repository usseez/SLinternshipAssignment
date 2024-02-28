import cv2
import sys
import numpy as np
import matplotlib.pylab as plt


def bgr_to_rgb(bgr):
    return bgr[:, :, ::-1]

def func(t):
    if t > 0.008856 :
        return (np.power(t, 1/3.0))
    else:
        return (7.787 * t + 16 / 116.0)




def thresh_rgb(var):
    # var1 = ((var0+0.055) / 1.055) ** 2.4
    # var2 = var0 / 12.92
    # var = np.where(var0 > 0.04045, var1, var2)
    
    
    if (var > 0.04045).any() :
        var = ((var+0.055) / 1.055) ** 2.4
    else:
        var = var / 12.92
        
    return var

def rgb2ciexyz(rgb):
    
    var_r = rgb[:,:,0]/255.0
    var_g = rgb[:,:,1]/255.0
    var_b = rgb[:,:,2]/255.0
    
    var_r = thresh_rgb(var_r) * 100.0
    var_g = thresh_rgb(var_g) * 100.0
    var_b = thresh_rgb(var_b) * 100.0
            
    xyz = [0, 0, 0]
    x = var_r * 0.4124 + var_g * 0.3576 + var_b * 0.1805
    y = var_r * 0.2126 + var_g * 0.7152 + var_b * 0.0722
    z = var_r * 0.0193 + var_g * 0.1192 + var_b * 0.9505
    
    
    # Observer= 2°, Illuminant= D65
    x = x / 95.047         # ref_X =  95.047
    y = y / 100.0          # ref_Y = 100.000
    z = z / 108.883
    
    xyz[0] = x.astype(np.uint8)
    xyz[1] = y.astype(np.uint8)
    xyz[2] = z.astype(np.uint8)
    
    ciexyz = cv2.merge((x,y,z))
    
    return ciexyz

def thresh_xyz(var):
    if (var > 0.008856).any() :
        var = (var) ** (1 / 3.0)
    else:
        var = 7.787 * var + (16.0 / 116.0)
    return var

def ciexyz2lab(ciexyz):
    
    var_x = ciexyz[:,:,0]
    var_y = ciexyz[:,:,1]
    var_z = ciexyz[:,:,2]
    
    var_x = thresh_xyz(var_x)
    var_y = thresh_xyz(var_y)
    var_z = thresh_xyz(var_z)

    l = ( 116.0 * var_y ) - 16.0
    a = 500.0 * ( var_x - var_y )
    b = 200.0 * ( var_y - var_z )
    
    l = l.astype(np.uint8)
    a = a.astype(np.uint8)
    b = b.astype(np.uint8)
    
    lab = cv2.merge((l,a,b))
    return lab




# BGR 이미지 불러오기
bgr_image = cv2.imread('cat.bmp')
height, width, channel = bgr_image.shape #불러온 이미지 크기 구하기
print(bgr_image)
# 불러왔는지 확인
if bgr_image is None:
    print('Image load failed!')
    sys.exit()
    
rgb = bgr_to_rgb(bgr_image)
print(rgb)
# BGR을 RGB로 변환
rgb_image = bgr_to_rgb(bgr_image)


# lab_image = rgb_to_lab(rgb_image)

# RGB를 ciexyz로 변환
ciexyz_image = rgb2ciexyz(rgb_image)

# ciexyz로 lab으로 변환
lab_image = ciexyz2lab(ciexyz_image)








# lab split
# print(lab_image.shape)
# l = lab_image[:,:,0]
# a = lab_image[:,:,1]
# b = lab_image[:,:,2]

# cv2.imshow('lab_image', lab_image)
# cv2.imshow('l', l)
# cv2.imshow('a', a)
# cv2.imshow('b', b)


restore_lab_plt = cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR)

restore_xyz_plt = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

# cv2.imshow('restore_lab_plt', restore_lab_plt)
# cv2.imshow('restore_xyz_plt', restore_xyz_plt)

# cv2.waitKey()