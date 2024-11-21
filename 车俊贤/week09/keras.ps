# 1.读取训练数据
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 2.打印训练图片
print("train_images:",train_images.shape)
# 3.打印训练标签
print("train_lables:",train_labels)
# 4.打印测试图像
print("test_images:",test_images.shape)
# 5.打印测试标签
print("test_images:",test_labels)
# 6.打开第一张图查看（测试图的第0张）
check = test_images[0]
import matplotlib.pyplot as plt

plt.imshow(check, cmap=plt.cm.binary)
plt.show()
from tensorflow.keras import models
from tensorflow.keras import layers

# 7.搭建一个空网络 models.Sequential()
network = models.Sequential()
# 8.给网络add()隐藏层,layers.Dense(层.全连接)也就是wx+b的接口，512隐藏层输出，激活函数activation='relu' ，输入节点个数（28x28）
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
# 9.给网络add()输出层，输出10个节点，十个数字，十个节点激活函数softmax
network.add(layers.Dense(10 , activation='softmax'))
#  10.优化网络optimizer='rmsprop'优化项
# loss='categorical_crossentropy' 损失函数
#  metrics=['accuracy'] 准确度计算方案
network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                metrics=['accuracy'])

# 11.把训练图片和测试图片都做成一维，然后归一化
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from tensorflow.keras.utils import to_categorical

# 12 打印one hot前第0个测试标签
print("origin test lable[0]:",test_labels[0])
# 13转换训练标签，to_categorical（）接口
train_labels = to_categorical(train_labels)
# 14转换测试标签，to_categorical（）接口
test_labels = to_categorical(test_labels)
# 15.输出更改后的标签
print("change test lables",test_labels[0])

# 16.用fit（）接口给搭建好的模型进行训练，训练图片，改后的训练标签
# batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
# epochs:每次计算的循环是五次
network.fit(train_images , train_labels ,epochs=5, batch_size=128)
# 17.阶段测试，输出损失函数 和成功率
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss)
print('test_acc', test_acc)

# 18 读图 测试图1号
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
check = test_images[1]
plt.imshow(check, cmap=plt.cm.binary)
plt.show()
# 19测试图集2维转1
test_images = test_images.reshape((10000, 28 * 28))
# 20结果 predict（）推理接口 推理测试图集
result = network.predict(test_images)

# 21.循环res中的每个数组，如果res当前的循环与one hot值对应 则输出这个数
for i in range(result[1].shape[0]):
    if (result[1][i] == 1):
        print("这个数字是",i,"吗？")
        break


