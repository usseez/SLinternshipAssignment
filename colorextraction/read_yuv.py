import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


width = 800
height = 1280

#golden_cat읽기
offset = int(width*height*3/2)
golden_frame = np.fromfile("golden_cat_yuv420.yuv",dtype='uint8')
Y_golden = golden_frame[0:width*height].reshape(height, width)
plt.imshow(Y_golden, cmap='gray')
plt.show()

#my_cat_yuv420읽기
frame = np.fromfile("yuv420_reduction.yuv",dtype='uint8')
print(frame.shape)
Y = frame[0:width*height].reshape(height, width)
plt.imshow(Y, cmap='gray')
plt.show()

print(golden_frame.dtype)
print(frame.dtype)


are_equal = np.array_equal(golden_frame, frame)

if are_equal:
    print("Both frames have identical pixel values.")
else:
    print("Frames have different pixel values.")

    # Find the differing pixels
    diff_locations = np.where(Y_golden != Y)

    # Plot the differing pixels
    plt.imshow(Y_golden, cmap='gray')
    plt.scatter(diff_locations[1], diff_locations[0], c='red', marker='x', label='Differing Pixels')
    plt.legend()
    plt.show()

    # print("Differing pixel locations:")
    # for y, x in zip(diff_locations[0], diff_locations[1]):
    #     print(f"({x}, {y}) - Golden: {Y_golden[y, x]}, Frame: {Y[y, x]}")

# golden_yuv = cv2.imread('golden_cat_yuv420.yuv')

# packed format은 access할 때 느려서 planar를 더 많이 사용함
# packed는 yuyv처럼 하나씩 jump해서 읽어야해
# planar는 yyyyy...uvuv이런 식으로 연속적으로 read함

# 알고리즘에 따라 처리속도가 달라 : 컴퓨터가 이해하려면 더 많은 명령어가 필요할 수 있음(packed>>planar)
# memset을쓰는 이유 : 초기화할 때 처리속도가 달라(zero로 하거나..... )
