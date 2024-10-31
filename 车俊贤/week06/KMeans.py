# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
#1.读取原始图像灰度颜色
img = cv2.imread('lenna.png',0)
print(img)
#2.获取图像高度、宽度
hi,wi = img.shape[:]
# 3.图像二维像素转换为一维
onedi = img.reshape((hi * wi , 1))
# 4.data表示聚类数据，最好是np.flloat32类型的N维点集
data = np.float32(onedi)
# 5.停止条件 (type,max_iter,epsilon)
# criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
# 其中，type有如下模式：
# —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
# —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
# —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TermCriteria_MAX_ITER, 10 , 1.0)
#6.设置标签，这种方式会随机选择初始的聚类中心。这种方法简单且易于实现，适用于数据分布未知的情况
flag = cv2.KMEANS_RANDOM_CENTERS
#7.在OpenCV中，Kmeans()函数原型如下所示：
# retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
# K-Means聚类 聚集成4类
# K表示聚类类簇数
# bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
# attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
# flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
# centers表示集群中心的输出矩阵，每个集群中心为一行数据
retval, labels, centers = cv2.kmeans(data, 4, None, criteria, 20, flag)
#8.重新生成最终图像
finimg = labels.reshape((img.shape[0], img.shape[1]))
#9.用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']
#10.显示图像
title = ['原始图像' , '聚类图像']
primg = [img , finimg]
for i in range(2):
    plt.subplot(1,2,i+1), plt.imshow(primg[i], 'gray')
    plt.title(title[i])
plt.show()
