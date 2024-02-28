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
start = time.time() # 시작 시간 저장
bgr565 = cv2.cvtColor(image, cv2.COLOR_BGR2BGR565)
print("time :", time.time() - start) # 현재시각 - 시작시간 = 실행 시간


