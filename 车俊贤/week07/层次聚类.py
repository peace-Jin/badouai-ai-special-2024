from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt
X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
Z = linkage(X, 'ward')
# 读取数据，并设置方案名
f = fcluster(Z,4,'distance')
# 计算数据，Z是数据，4是限定的循环次数，distance是以距离的方式算层次聚类
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
plt.show()
