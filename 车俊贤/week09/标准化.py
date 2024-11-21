import numpy as np
import matplotlib.pyplot as plt
#归一化的两种方式
def N1(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]
    '''归一化（0~1）'''
    '''x_=(x−x_min)/(x_max−x_min)'''
def N2(x):
    return [(float(i)-np.mean(x))/float(max(x) - min(x)) for i in x]

    '''归一化（-1~1）'''
    '''x_=(x−x_mean)/(x_max−x_min)'''

#标准化
def zs(x):
    '''x∗=(x−μ)/σ'''
    x_mean = np.mean(x)
    s2 = sum([(i - np.mean(x)) * (i - np.mean(x)) for i in x]) / len(x)
    return [(i - x_mean) / s2 for i in x]

l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1=[]
# for i in l:
#     i+=2
#     l1.append(i)
# print(l1)
cs = []
for i in l:
     c= l.count(i)
     cs.append(c)
print(cs)
n2 = N2(l)
n1 = N1(l)
z = zs(l)
print(n1)
print(n2)
print(z)
'''
蓝线为原始数据，橙线为z
'''
plt.plot(l,cs)
plt.plot(z,cs)
plt.show()
