import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

if __name__ =='__main__':
  img=plt.imread('lenna.png')
  file='lenna.png'
  print('ori',img)
  if file[-4:]=='.png':
    img=img*255
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  #高斯平滑
  sigma=0.5
  core=5
  goshcore=np.zeros([core,core])
  tmp=[i-core//2 for i in range(core)]
  m1=1/(2*math.pi*sigma**2)
  m2=-1/(2*sigma**2)
  for i in range(core):
      for j in range(core):
          goshcore[i,j]=m1*math.exp(m2*(tmp[i]**2+tmp[j]**2))
  goshcore=goshcore/goshcore.sum()
  dx,dy=img.shape
  goshimg=np.zeros(img.shape)
  tmp=core//2
  img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')
  for i in range(dx):
      for j in range(dy):
          goshimg[i, j] = np.sum(img_pad[i:i + core, j:j + core] * goshcore)
  plt.figure(1)
  plt.imshow(goshimg.astype(np.uint8), cmap='gray')
  plt.axis('off')

  #卷积
  sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
  img_tidu_x = np.zeros(goshimg.shape)
  img_tidu_y = np.zeros([dx, dy])
  img_tidu = np.zeros(goshimg.shape)
  img_pad = np.pad(goshimg, ((1, 1), (1, 1)), 'constant')
  for i in range(dx):
      for j in range(dy):
          img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)
          img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
          img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
  img_tidu_x[img_tidu_x == 0] = 0.00000001
  angle = img_tidu_y / img_tidu_x
  plt.figure(2)
  plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
  plt.axis('off')

  # 非极大值抑制
  img_yizhi = np.zeros(img_tidu.shape)
  for i in range(1, dx - 1):
    for j in range(1, dy - 1):
      flag = True
      temp = img_tidu[i - 1:i + 2, j - 1:j + 2]
      if angle[i, j] <= -1:
        num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
        num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
        if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
          flag = False
      elif angle[i, j] >= 1:
        num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
        num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
        if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
          flag = False
      elif angle[i, j] > 0:
        num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
        num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
        if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
          flag = False
      elif angle[i, j] < 0:
        num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
        num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
        if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
          flag = False
      if flag:
        img_yizhi[i, j] = img_tidu[i, j]
  plt.figure(3)
  plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
  plt.axis('off')


  lower_boundary = img_tidu.mean() * 0.5
  high_boundary = lower_boundary * 3
  zhan = []
  for i in range(1, img_yizhi.shape[0] - 1):
    for j in range(1, img_yizhi.shape[1] - 1):
      if img_yizhi[i, j] >= high_boundary:
        img_yizhi[i, j] = 255
        zhan.append([i, j])
      elif img_yizhi[i, j] <= lower_boundary:
        img_yizhi[i, j] = 0

  while not len(zhan) == 0:
    temp_1, temp_2 = zhan.pop()
    a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
    if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
      img_yizhi[temp_1 - 1, temp_2 - 1] = 255
      zhan.append([temp_1 - 1, temp_2 - 1])
    if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
      img_yizhi[temp_1 - 1, temp_2] = 255
      zhan.append([temp_1 - 1, temp_2])
    if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
      img_yizhi[temp_1 - 1, temp_2 + 1] = 255
      zhan.append([temp_1 - 1, temp_2 + 1])
    if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
      img_yizhi[temp_1, temp_2 - 1] = 255
      zhan.append([temp_1, temp_2 - 1])
    if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
      img_yizhi[temp_1, temp_2 + 1] = 255
      zhan.append([temp_1, temp_2 + 1])
    if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
      img_yizhi[temp_1 + 1, temp_2 - 1] = 255
      zhan.append([temp_1 + 1, temp_2 - 1])
    if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
      img_yizhi[temp_1 + 1, temp_2] = 255
      zhan.append([temp_1 + 1, temp_2])
    if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
      img_yizhi[temp_1 + 1, temp_2 + 1] = 255
      zhan.append([temp_1 + 1, temp_2 + 1])

  for i in range(img_yizhi.shape[0]):
    for j in range(img_yizhi.shape[1]):
      if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
        img_yizhi[i, j] = 0


  plt.figure(4)
  plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
  plt.axis('off')
  plt.show()
