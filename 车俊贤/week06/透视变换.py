import cv2
import numpy as np

# 读图
img = cv2.imread('photo1.jpg')
# 调用拷贝
copyimg = img.copy()
'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
# 打印原图shape
print(img.shape)
# 生成透视变换矩阵；进行透视变换 cv2.getPerspectiveTransform（原图矩阵，结果矩阵）
warp = cv2.getPerspectiveTransform(src,dst)
print("warpMatrix:")
# 打印warp
print(warp)
# cv2.warpPerspective（拷贝图，变换矩阵，（图片大小））
result = cv2.warpPerspective(copyimg , warp , (337,488))
# cvshow 原图、结果图
cv2.imshow("src",img)
cv2.imshow("res",result)
cv2.waitKey(0)
