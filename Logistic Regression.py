### Sigmoid Func ###
def sigmoidFunc(data):
    g = 1.0 / ( 1.0 + np.exp(-data))
    return g


### Training Logistic Func ###
def trainLogistic(dataset):
    data = dataset

    # 注意这里读入的矩阵默认是既没有行标签，也没有列标签的

    # 标注股票是涨还是跌的数据点，1代表涨，0代表跌
    # 我们的数据集第三列是标注
    judgeTag = list()
    x,y = np.shape(data)
    for j in range(x):
        judgeTag.append(data[j][2])
    judgeTag = np.transpose(np.mat(judgeTag))
    #print(judgeTag)
    # 从第4列开始是特征数据，不从第3列开始是因为第三列是价格，我们的涨跌本身就是从这个数据上得来的，已经得到过一次信息了
    # 权重w的更新使用随机梯度下降，步长是alpha，我们训练的目标是不断的更新权重。
    n = 87 # 这里定义为87是因为我们已经知道特征有87个
    weights = np.ones((n,1)) # 创建 n 行，一列的列向量
    alpha = 0.001 # 下降的步长
    cycles = 500 # 循环次数
    sampleMatrix = data[:,3:] #样本矩阵

    ### 以上为常量 ###

    ### 一下为每训练一次都会更新一次的量 ###
    for k in range(n):
        # 结果矩阵
        h = sigmoidFunc(sampleMatrix.dot(weights))
        #print(np.shape(sampleMatrix.dot(weights)))
        error = judgeTag - h
        #print(np.shape(error))
        transSampleMatrix = np.transpose(sampleMatrix)
        #print(np.shape(transSampleMatrix))
        weights = weights + (alpha * transSampleMatrix.dot(error))
        #print(np.shape(weights))
   
    return weights


### Test Error function ###
def test(data,w):
    judgeTag = list()
    x,y = np.shape(data)
    for j in range(x):
        judgeTag.append(data[j][2])
    judgeTag = np.transpose(np.mat(judgeTag))
    sampleMatrix = data[:,3:]
    
#     correctlist = list()
#     x,y = np.shape(judgeTag)
#     for i in range(y):
    error = judgeTag - sigmoidFunc(sampleMatrix.dot(w))
    error = np.square(error)
    #print(judgeTag)    
    return np.mean(error)
