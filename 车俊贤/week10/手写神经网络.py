import numpy
import scipy.special

class NeuralNetWork:
    def __init__(self,inputnodes,hiddennodes,finalouts,learnrate):
        # 1.构造函数，初始化网络，i设置输入层，h中间层，和o输出层节点数
        self.innodes = inputnodes
        self.hnodes = hiddennodes
        self.fnodes = finalouts
        # 2.构造学习率lr
        self.lr = learnrate
        # 3.隐藏层输入权重：
        # （（随机高斯分布设置：均值0.0，标准差：隐藏层节点的-1/2次方），（输出矩阵：以隐藏层节点数，输入层节点数输出））
        # mean：正态分布的均值，生成的随机数会围绕这个值波动。
        # std：标准差，控制生成随机数的分散程度，标准差越大，生成的数值越分散；标准差越小，数值则越集中。
        # size：以隐藏层节点数，输入层节点数输出
        self.whi = (numpy.random.normal(0.0,pow(self.hnodes ,-0.5),(self.hnodes,self.innodes)))
        # 4.隐藏层输出权重：
        # （（随机高斯分布设置：均值0.0，标准差：输出层节点的-1/2次方），（输出矩阵：以输出层节点数，隐藏层节点数输出））
        self.who = (numpy.random.normal(0.0, pow(self.fnodes, -0.5), (self.fnodes, self.hnodes)))
        # 5.激活函数 activation_function=lambda x:scipy.special.expit(x)
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # 6训练函数 输入列，目标列
    def train(self,inputlist , targetlist):
        '''
        7把inputs_list, targets_list转换成numpy支持的二维矩阵
        .T表示做矩阵的转置
        '''
        inputs = numpy.array(inputlist,ndmin = 2).T
        targets = numpy.array(targetlist, ndmin=2).T
        # 8计算信号经过输入层后产生的信号量
        hinput = numpy.dot(self.whi , inputs)
        # 9中间层神经元对输入的信号做激活函数后得到输出信号
        hout = self.activation_function(hinput)
        # 10输出层接收来自中间层的信号量
        oinput = numpy.dot(self.who , hout)
        # 11 输出层对信号量进行激活函数后得到最终输出信号
        oout = self.activation_function(oinput)
        # 12计算误差
        output_errors = targets - oout
        hidden_errors = numpy.dot(self.who.T, output_errors *oout * (1 - oout))
        # 13根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr * numpy.dot((output_errors * oout * (1 - oout)),
                                        numpy.transpose(hout))
        self.whi += self.lr * numpy.dot((hidden_errors * hout * (1 - hout)),
                                        numpy.transpose(inputs))
        pass

    # 14输出函数 inputs
    def defoutput(self,inputs):
        # 根据输入数据计算并输出答案
        # 15计算中间层从输入层接收到的信号量
        hinput = numpy.dot(self.whi, inputs)
        # 16计算中间层经过激活函数后形成的输出信号量
        hout = self.activation_function(hinput)
        # 17计算最外层接收到的信号量
        oinput = numpy.dot(self.who, hout)
        # 18计算最外层神经元经过激活函数后输出的信号量
        oout = self.activation_function(oinput)
        # 19打印结果
        print(oout)
        return oout

# 20初始化网络
'''
由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点
'''
# 21设置值 输入层 隐藏层200 输出层 10 学习率10
inputnodes = 784
hiddennodes = 200
outputnodes = 10
learningdata = 0.1
# 22给网络类传参
n = NeuralNetWork(inputnodes,hiddennodes,outputnodes,learningdata)
# 23读入训练数据
# open函数里的路径根据数据存储的路径来设定
training_data_file = open("dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
# 24加入epocs,设定网络的训练循环次数，共5行数据
epochs = 5
for e in range(epochs):
    # 25以训练数据循环
    for i in training_data_list:
        # 26把数据依靠','区分，并分别读入
        all_values = i.split(',')
        # 27 归一化
        inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        # 设置图片与数值的对应关系
        # 28以输出节点的数量做目标矩阵 归零后加0.01
        target = numpy.zeros(outputnodes)+0.01
        # 29取当前循环中 当前数据行的第一个标签值作为标准值 设置为0.99
        target[int(all_values[0])] = 0.99
        # 30 输入数据和标准值传入训练函数
        n.train(inputs,target)

# 31读取测试数据
test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
# 32以测试数据循环
for i in test_data_list:
    # 33逗号分隔
    all_values = i.split(',')
    # 34正确数字为标签值
    correct_number = int(all_values[0])
    print("该图片对应的数字为:", correct_number)
    # 35归一化
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    # 36输出为给query函数传参inputs后的返回值
    output = n.defoutput(inputs)
    # 37找到数值最大的神经元对应的编号 numpy.argmax（）
    label = numpy.argmax(output)
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)
# 计算图片判断的成功率
scores_array = numpy.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)
