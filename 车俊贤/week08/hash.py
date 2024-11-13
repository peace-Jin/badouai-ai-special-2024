import cv2
import numpy as np

# 均值哈希算法
def ahash(img):
    img=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str= ' '
    avg = np.mean(gray)
    for i in range(8):
        for j in range(8):
            if gray[i,j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

# 差值算法
def dHash(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str= ''
    for i in range(8):
        for j in range(8):
            if gray[i,j] > gray[i,j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

# Hash值对比
def chash(ahash,dhash):
    if len(ahash) !=len(dhash):
        print("传参错误")
        return -1
    n=0
    for i in range(len(ahash)):
        if ahash[i] != dhash[i]:
            n+=1
    return n

oriimg = cv2.imread("lenna.png")
noiimg = cv2.imread("lenna_noise.png")
hash1 = ahash(oriimg)
hash2 = ahash(noiimg)
print("原图均值哈希值：",hash1)
print("噪声图均值哈希值：",hash2)
n=chash(hash1,hash2)
print("均值哈希算法相似度：",n)

hash1 = dHash(oriimg)
hash2 = dHash(noiimg)
print("原图差值哈希值：",hash1)
print("噪声图差值哈希值：",hash2)
n=chash(hash1,hash2)
print("差值哈希算法相似度：",n)


