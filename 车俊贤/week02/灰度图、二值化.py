# -*- coding: utf-8 -*-
@author: Cjx
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 未使用cv2

# 1.灰度图
img=plt.imread("lenna.png")
plt.subplot(221)
plt.imshow(img)
img_gray=rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray , cmap='gray')

# 2.二值化
row,col=img_gray.shape
for i in range(row):
    for j in range(col):
        if img_gray[i,j]<=0.5:
            img_gray[i,j]=0
        else:
            img_gray[i,j]=1
plt.subplot(223)
plt.imshow(img_gray,cmap='gray')
plt.show()
