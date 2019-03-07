import numpy
import pandas
import math,copy,scipy.io

num_feature = 400
batch_size = 500
EpochNum = 500
learning_rate = 1000
NeuronNum_list = [num_feature,25,26,10]

trainingSet_Size = 3000
validitySet_Size = 500
testSet_Size = 1500
NumPic_path = r'.\NumPicData.mat'

class NumPicData(object):
    # 初始化数据集对象
    def __init__(self,trainingData,trainingLabel,ValidatyData,ValidatyLabel,TestData,TestLabel):
        # Load data set
        # The Feature Set should be SampleNum * FeatureNum
        # The Label Set should be SampleNum * 1
        self.tempIndex = 0
        self.currIndex = 0
        self.readout = False
        self.TrainingFeature = trainingData
        self.TrainingLabel = trainingLabel
        self.ValidityFeature = ValidatyData
        self.ValidityLabel = ValidatyLabel
        self.TestFeature = TestData
        self.TestLabel = TestLabel
        self.TrainingDataLen = self.TrainingFeature.shape[0]

    def get_TrainingSet(self):
        return self.TrainingFeature,self.TrainingLabel

    def get_ValiditySet(self):
        return self.ValidityFeature,self.ValidityLabel

    def get_TestSet(self):
        return self.TestFeature,self.TestLabel

    def init_currIndex(self):
        self.currIndex = 0
        self.readout = False

    # 定义获取batch方法
    def get_NextTrainingBatch(self,batchSize):
        # print(self.tempIndex)
        # print(self.currIndex)
        if self.currIndex + batchSize < self.TrainingDataLen:
            self.tempIndex = self.currIndex
            self.currIndex = self.currIndex + batchSize
            return self.TrainingFeature[self.tempIndex:self.currIndex,:], self.TrainingLabel[self.tempIndex:self.currIndex,:],self.readout
        else:
            self.tempIndex = self.currIndex
            self.currIndex = 0
            self.readout = True
            return self.TrainingFeature[self.tempIndex:self.TrainingDataLen,:], self.TrainingLabel[self.tempIndex:self.TrainingDataLen,:],self.readout

# 读取训练数据【使用吴恩达老师课程的训练数据，是手写数字识别】
def read_data(path,trainingSet_Size,validitySet_Size,testSet_Size):
    # 导入.mat数据文件
    NumPics_data = scipy.io.loadmat(path)
    # X是feature数据  y是label数据
    data = NumPics_data['X']
    label_orig = NumPics_data['y']
    len_data = data.shape[0]

    # 源数据都是按舒徐排列的，现在需要进行打乱操作
    orig_indexList = list(range(len_data))
    numpy.random.shuffle(orig_indexList)

    # 使用相同的随机序列来打乱数据集【保证feature和label任然意义对应】
    data = data[orig_indexList]
    label_orig = label_orig[orig_indexList]
    # print(label_orig)

    # 生成one-hot类型的label，把原来的1，2，3，4...的数据变为[0 1 0 ... 0 0] [0 0 1 0 ... 0 0]类型的label
    new_label_shape = (label_orig.shape[0],10)
    # 新的label
    label = numpy.zeros(new_label_shape)

    for i in range(label_orig.shape[0]):
        value_label = int(label_orig[i]%10)
        label[i,value_label] = 1
    # print(label)
    # print(data)
    # print(type(data))
    # print(data.shape)
    # print(data.dtype)
    # print(data.shape[0])
    # print(data.shape[1])
    # print(type(data.shape[0]))
    # print(type(data.shape[1]))

    # 按照既定比例进提取各个集合
    trainingData = data[0:trainingSet_Size,:]
    extra_ones = numpy.ones((trainingData.shape[0],1))
    trainingData = numpy.hstack((extra_ones,trainingData))

    trainingLabel = label[0:trainingSet_Size,:]
    # trainingLabel = trainingLabel.reshape(len(trainingLabel),1)


    ValidatyData = data[trainingSet_Size:trainingSet_Size+validitySet_Size,:]
    extra_ones = numpy.ones((ValidatyData.shape[0],1))
    ValidatyData = numpy.hstack((extra_ones,ValidatyData))

    ValidatyLabel = label[trainingSet_Size:trainingSet_Size+validitySet_Size,:]
    # ValidatyLabel = ValidatyLabel.reshape(len(ValidatyLabel),1)


    TestData = data[trainingSet_Size+validitySet_Size:trainingSet_Size+validitySet_Size+testSet_Size,:]
    extra_ones = numpy.ones((TestData.shape[0],1))
    TestData = numpy.hstack((extra_ones,TestData))

    TestLabel = label[trainingSet_Size+validitySet_Size:trainingSet_Size+validitySet_Size+testSet_Size,:]
    # TestLabel = TestLabel.reshape(len(TestLabel),1)

    allNumPicData = NumPicData(trainingData,trainingLabel,ValidatyData,ValidatyLabel,TestData,TestLabel)

    # 返回数据集对象
    return allNumPicData

# 增加一列全1数据【偏置列】
def add_extraBias(data):
    sampleNum = data.shape[0]
    extra_ones = numpy.ones((sampleNum,1))
    newdata = numpy.hstack((extra_ones,data))
    return newdata

# 计算sigmoid函数
def sigmoid(input_NumVec):
    # numpy报错问题，需要进行一步.astype才能继续进行
    input_NumVec = (-input_NumVec).astype(float)

    sigmoid_value = 1/(1+numpy.exp(input_NumVec))
    return sigmoid_value

def test_data(path,trainingSet_Size,validitySet_Size,testSet_Size):
    # 测试导入的数据是否正常
    # Test input data
    allNumPicData = read_data(path,trainingSet_Size,validitySet_Size,testSet_Size)
    trainingData,trainingLabel = allNumPicData.get_TrainingSet()
    ValidatyData,ValidatyLabel = allNumPicData.get_ValiditySet()
    TestData,TestLabel = allNumPicData.get_TestSet()
    print(trainingData.shape)
    print(ValidatyData.shape)
    print(TestData.shape)

    print(trainingLabel.shape)
    print(ValidatyLabel.shape)
    print(TestLabel.shape)

    print(TestData)
    # Test sigmoid function
    # sigmoid_value = sigmoid(TestLabel)
    # print(sigmoid_value)
    # print(sigmoid_value.shape)
    #
    # print(len(TestLabel))
    # print(len(TestData))

# 单层前向传播函数
def forw_propagation(input_mat,theta):
    input_currLayer = numpy.dot(input_mat,theta)
    output_currLayer = sigmoid(input_currLayer)
    return input_currLayer,output_currLayer

# 计算代价函数【使用cross entropy代价函数】
def calc_J(prediction,label):
    num_sample = len(prediction)
    J = -1/num_sample*(label*numpy.log(prediction)+(1-label)*numpy.log(1-prediction)).sum()
    return J

# 误差反向传播函数
def sigma_BackPropagation(delta_nextLayer,theta_currLayer,output_currLayer):
    sigma_currLayer = numpy.dot(delta_nextLayer,theta_currLayer.T) * (output_currLayer*(1-output_currLayer))
    return sigma_currLayer

# 计算梯度
def calc_gradient(output_currLayer,delta_nextLayer):
    num_sample = output_currLayer.shape[0]
    theta_gradient = numpy.dot(output_currLayer.T,delta_nextLayer)
    theta_gradient = theta_gradient/num_sample
    return theta_gradient


class NeuralNetwork(object):
    # 定义神经网络对象【需要各层神经元节点数和数据集对象】
    def __init__(self,NeuronNum_list,allNumPicData):
        self.allNumPicData = allNumPicData
        # len(NeuronNum_list) must be 4 [including output layer and input layer]
        self.theta_inputLayer = numpy.random.random((NeuronNum_list[0]+1,NeuronNum_list[1]))
        self.theta_hidenLayer1 = numpy.random.random((NeuronNum_list[1]+1,NeuronNum_list[2]))
        self.theta_hidenLayer2 = numpy.random.random((NeuronNum_list[2]+1,NeuronNum_list[3]))

    # 逐层进行前向传播
    def FordProp(self,feature_currBatch,theta_inputLayer,theta_hidenLayer1,theta_hidenLayer2):
        # PROPAGATION TO HIDEN LAYER 1
        self.activedValue_hidenLayer1,self.outputvalue_hidenLayer1 = forw_propagation(feature_currBatch,theta_inputLayer)
        '''
        INPUT: batch_size*(num_feature+1) DOT_* (num_feature+1)*NumNeural_hidenLayer1 
        OUTPUT: batch_size*NumNeural_hidenLayer1
        '''

        # PROPAGATION TO HIDEN LAYER 2
        self.inputvalue_hidenLayer2 = add_extraBias(self.outputvalue_hidenLayer1)
        '''EXTEND THE VALUE WITH ONE MORE CONSTANT "1" '''
        self.activedValue_hidenLayer2,self.outputvalue_hidenLayer2 = forw_propagation(self.inputvalue_hidenLayer2,theta_hidenLayer1)
        '''
        INPUT: batch_size*(NumNeural_hidenLayer1+1) DOT_* (NumNeural_hidenLayer1+1)*NumNeural_hidenLayer2
        OUTPUT: batch_size*NumNeural_hidenLayer2
        '''

        # PROPAGATION TO OUTPUT LAYER
        self.inputvalue_outputLayer = add_extraBias(self.outputvalue_hidenLayer2)
        '''EXTEND THE VALUE WITH ONE MORE CONSTANT "1" '''
        self.activedValue_outputLayer,self.outputvalue_outputLayer = forw_propagation(self.inputvalue_outputLayer,theta_hidenLayer2)
        '''
        INPUT: batch_size*(NumNeural_hidenLayer2+1) DOT_* (NumNeural_hidenLayer2+1)*NumNeural_outputLayer
        OUTPUT: batch_size*NumNeural_outputLayer
        '''
        return self.outputvalue_outputLayer

    # 计算代价函数
    def CalcCostFunc(self,output,label):
        # CALC COST FUNCTION VALUE
        costFuncValue = calc_J(output,label)
        '''
        INPUT: batch_size*NumNeural_outputLayer    batch_size*NumNeural_outputLayer
        OUTPUT: SCALAR-COST FUNCTION VALUE 
        '''
        return costFuncValue

    # 反向传播误差值
    def BackPropDelta(self,theta_hidenLayer2,theta_hidenLayer1,outputvalue_outputLayer,label_currBatch):
        # CALC THE DELTA IN OUTPUT LAYER
        self.delta_output = outputvalue_outputLayer - label_currBatch
        '''
        GOT: batch_size*(NumNeural_outputLayer)
        '''

        # DELTA BACK PROPAGATION TO HIDEN LAYER 2
        self.delta_hidenLayer2 = sigma_BackPropagation(self.delta_output,
                                                       theta_hidenLayer2,
                                                       self.inputvalue_outputLayer)
        self.delta_hidenLayer2 = self.delta_hidenLayer2[:,1:]
        '''
        INPUT: batch_size*NumNeural_outputLayer DOT_* [(NumNeural_hidenLayer2+1)*NumNeural_outputLayer].T ** batch_size*(NumNeural_hidenLayer2+1)
        OUTPUT:batch_size*NumNeural_hidenLayer2
        '''

        # DELTA BACK PROPAGATION TO HIDEN LAYER 1
        self.delta_hidenLayer1 = sigma_BackPropagation(self.delta_hidenLayer2,
                                                       theta_hidenLayer1,
                                                       self.inputvalue_hidenLayer2)
        self.delta_hidenLayer1 = self.delta_hidenLayer1[:,1:]
        '''[** MEANS 点乘]
        INPUT: batch_size*NumNeural_hidenLayer2 DOT_* [(NumNeural_hidenLayer1+1)*NumNeural_hidenLayer2].T ** batch_size*(NumNeural_hidenLayer1+1)
        OUTPUT:batch_size*NumNeural_hidenLayer1
        '''

    # 根据计算的误差值计算theta梯度
    def CalcGradient(self,feature_currBatch):
        # CALC THE GRADIENT FOR THETA_INPUTLAYER [VALUE HAS BEEN DIVIDED BY SAMPLE NUM]
        self.theta_inputLayer_Gradient = calc_gradient(feature_currBatch,self.delta_hidenLayer1)
        '''
        INPUT: [batch_size*(num_feature+1)].T DOT_* batch_size*NumNeural_hidenLayer1
        OUTPUT: (num_feature+1)*NumNeural_hidenLayer1
        '''

        # CALC THE GRADIENT FOR THETA_HIDENLAYER 1 [VALUE HAS BEEN DIVIDED BY SAMPLE NUM]
        self.theta_hidenLayer1_Gradient = calc_gradient(self.inputvalue_hidenLayer2,self.delta_hidenLayer2)
        '''
        INPUT: [batch_size*(NumNeural_hidenLayer1+1)].T DOT_* batch_size*NumNeural_hidenLayer2
        OUTPUT: (NumNeural_hidenLayer1+1)*NumNeural_hidenLayer2
        '''

        # CALC THE GRADIENT FOR THETA_HIDENLAYER 2 [VALUE HAS BEEN DIVIDED BY SAMPLE NUM]
        self.theta_hidenLayer2_Gradient = calc_gradient(self.inputvalue_outputLayer,self.delta_output)
        '''
        INPUT: [batch_size*(NumNeural_hidenLayer2+1)].T DOT_* batch_size*NumNeural_outputLayer
        OUTPUT: (NumNeural_hidenLayer2+1)*NumNeural_outputLayer
        '''

    # 梯度检查-计算数值型误差
    def calc_NumericalGradient(self,feature_currBatch,label_checkBatch):
        epcelen = 0.001
        gradient_checkresult = numpy.zeros(self.theta_inputLayer.shape)
        for i in range(self.theta_inputLayer.shape[0]):
            for j in range(self.theta_inputLayer.shape[1]):
                theta_inputLayer_temp_max = copy.deepcopy(self.theta_inputLayer)
                theta_inputLayer_temp_min = copy.deepcopy(self.theta_inputLayer)
                theta_inputLayer_temp_max[i,j] = theta_inputLayer_temp_max[i,j] + epcelen
                theta_inputLayer_temp_min[i,j] = theta_inputLayer_temp_min[i,j] - epcelen
                outputLayerValue_max = self.FordProp(feature_currBatch,theta_inputLayer_temp_max,self.theta_hidenLayer1,self.theta_hidenLayer2)
                outputLayerValue_min = self.FordProp(feature_currBatch,theta_inputLayer_temp_min,self.theta_hidenLayer1,self.theta_hidenLayer2)
                costFuncValue_max =  self.CalcCostFunc(outputLayerValue_max,label_checkBatch)
                costFuncValue_min =  self.CalcCostFunc(outputLayerValue_min,label_checkBatch)
                gradient_checkresult[i,j] = (costFuncValue_max-costFuncValue_min)/(2*epcelen)
        return gradient_checkresult

    # 梯度检验
    def check_gradient(self):
        feature_checkBatch,label_checkBatch = self.allNumPicData.get_TestSet()
        outputvalue_outputLayer = self.FordProp(feature_checkBatch,self.theta_inputLayer,self.theta_hidenLayer1,self.theta_hidenLayer2)
        self.currCostFuncValue =  self.CalcCostFunc(self.outputvalue_outputLayer,label_checkBatch)
        self.BackPropDelta(self.theta_hidenLayer2,self.theta_hidenLayer1,outputvalue_outputLayer,label_checkBatch)
        self.CalcGradient(feature_checkBatch)
        gradient_checkresult_inputLayer = self.calc_NumericalGradient(feature_checkBatch,label_checkBatch)
        diff_gradient_inputLayer = gradient_checkresult_inputLayer - self.theta_inputLayer_Gradient
        relat_diff = numpy.abs(diff_gradient_inputLayer).sum()/numpy.abs(gradient_checkresult_inputLayer).sum()
        print('反向传播计算所得input layer梯度：')
        print(self.theta_inputLayer_Gradient)
        print('*'*100)
        print('数值型计算所得input layer梯度：')
        print(gradient_checkresult_inputLayer)
        print('*'*100)
        print('两种方法计算所得input layer梯度差值：')
        print(diff_gradient_inputLayer)
        print('*'*100)
        print('\n\t梯度检查差异: %f\n'%relat_diff)

    # 更新theta
    def updateTheta(self,learning_rate):
        # UPDATE THETAS
        # print(self.theta_hidenLayer1)
        # print(learning_rate*self.theta_hidenLayer1_Gradient)
        self.theta_inputLayer = self.theta_inputLayer - learning_rate*self.theta_inputLayer_Gradient
        self.theta_hidenLayer1 = self.theta_hidenLayer1 - learning_rate*self.theta_hidenLayer1_Gradient
        self.theta_hidenLayer2 = self.theta_hidenLayer2 - learning_rate*self.theta_hidenLayer2_Gradient
        # print(self.theta_hidenLayer1)
        # print('THETA UPDATED!')

    # 计算precesion【此处不用这个检查】
    def calc_Precesion(self,output,label):
        got_positive = output * label
        if output.sum() == 0:
            return 0
        else:
            precesion = got_positive.sum()/output.sum()
        return precesion

    # 计算recall【此处不用这个检查】
    def calc_Recall(self,output,label):
        got_positive = output * label
        if label.sum() == 0:
            return 0
        else:
            recall = got_positive.sum()/label.sum()
        return recall

    # 计算准确率
    def calc_accuracy(self,output,label):
        argmax_output = numpy.argmax(output,axis=1)
        argmax_label = numpy.argmax(label,axis=1)
        accuracy = numpy.equal(argmax_output,argmax_label).sum()/label.shape[0]
        return accuracy

    # 训练神经网络主函数
    def trainNeuralNetwork(self,batch_size,learning_rate,EpochNum):
        for i in range(EpochNum):
            i_sample = 0
            # print(self.allNumPicData.readout)
            learning_rate_curr = learning_rate/(i+1)
            while not self.allNumPicData.readout:
                # 定期输出结果
                if (i_sample % 500) == 0 and i_sample !=0:
                    print('CostFunc Value After %d Samples Training: %.2f [Epic %d]'%(i_sample,self.currCostFuncValue,i+1))

                    if (i_sample % 2000) == 0:
                        # 每20000此进行验证和测试
                        self.validity_model()
                        print('Validity Set Check:')
                        print('\t[CostFuncValue] After %d Samples on ValiditySet: %.2f [Epic %d]'%(i_sample,self.costFuncValue_validity,i+1))
                        print('\t[Accuracy] After %d Samples on ValiditySet: %.2f%% [Epic %d]'%(i_sample,self.valisity_accuracy*100,i+1))
                        print('\t[Precesion] After %d Samples on ValiditySet: %.2f%% [Epic %d]'%(i_sample,self.valisity_precrsion*100,i+1))
                        print('\t[Recall] Value After %d Samples on ValiditySet: %.2f%% [Epic %d]'%(i_sample,self.valisity_recall*100,i+1))

                        self.test_model()
                        print('Test Set Check:')
                        print('\t[CostFuncValue] After %d Samples on TestSet: %.2f [Epic %d]'%(i_sample,self.costFuncValue_test,i+1))
                        print('\t[Accuracy] After %d Samples on TestSet: %.2f%% [Epic %d]'%(i_sample,self.test_accuracy*100,i+1))
                        print('\t[Precesion] After %d Samples on TestSet: %.2f%% [Epic %d]'%(i_sample,self.test_precrsion*100,i+1))
                        print('\t[Recall] Value After %d Samples on TestSet: %.2f%% [Epic %d]'%(i_sample,self.test_recall*100,i+1))
                # GOT THE VALUE IN INPUT LAYER
                # 获取当前数据集
                self.feature_currBatch,self.label_currBatch,_ = self.allNumPicData.get_NextTrainingBatch(batch_size)
                '''
                GOT: batch_size*(num_feature+1)
                NO NEED TO ADD EXTRA ONE
                '''
                # 前向传播
                self.outputvalue_outputLayer = self.FordProp(self.feature_currBatch,self.theta_inputLayer,self.theta_hidenLayer1,self.theta_hidenLayer2)
                # 计算代价函数
                self.currCostFuncValue =  self.CalcCostFunc(self.outputvalue_outputLayer,self.label_currBatch)
                # 反向传播误差
                self.BackPropDelta(self.theta_hidenLayer2,self.theta_hidenLayer1,self.outputvalue_outputLayer,self.label_currBatch)
                # 计算梯度
                self.CalcGradient(self.feature_currBatch)
                # 更新theta
                self.updateTheta(learning_rate_curr)
                i_sample += batch_size
            # 数据集遍历一次后初始化index，从头开始获取batch
            self.allNumPicData.init_currIndex()

    def validity_model(self):
        # 测试训练好的模型在检验集上的效果
        # 检验过程会影响网络中各层神经元的输入和输出值，不会影响theta
        self.validitySet,self.validityLabel = self.allNumPicData.get_ValiditySet()
        validityOutput = self.FordProp(self.validitySet,self.theta_inputLayer,self.theta_hidenLayer1,self.theta_hidenLayer2)
        self.costFuncValue_validity = self.CalcCostFunc(validityOutput,self.validityLabel)
        # 进行整数化，方便对比【之前做二分类用的，在数字识别里取消】
        # validityOutput = numpy.round(validityOutput)
        # CALC THE ACCURACY/PRECESION/RECALL
        self.valisity_accuracy = self.calc_accuracy(validityOutput,self.validityLabel)
        self.valisity_precrsion = self.calc_Precesion(validityOutput,self.validityLabel)
        self.valisity_recall = self.calc_Recall(validityOutput,self.validityLabel)

    def test_model(self):
        # 测试训练好的模型在测试集上的效果
        # 检验过程会影响网络中各层神经元的输入和输出值，不会影响theta
        self.testSet,self.testLabel = self.allNumPicData.get_TestSet()
        # print(self.testSet)
        testOutput = self.FordProp(self.testSet,self.theta_inputLayer,self.theta_hidenLayer1,self.theta_hidenLayer2)
        self.costFuncValue_test = self.CalcCostFunc(testOutput,self.testLabel)
        # 进行整数化，方便对比【之前做二分类用的，在数字识别里取消】
        # testOutput = numpy.round(testOutput)
        # CALC THE ACCURACY/PRECESION/RECALL
        print(testOutput)
        print(numpy.argmax(testOutput,axis =1))
        print(testOutput.shape)
        self.test_accuracy = self.calc_accuracy(testOutput,self.testLabel)
        self.test_precrsion = self.calc_Precesion(testOutput,self.testLabel)
        self.test_recall = self.calc_Recall(testOutput,self.testLabel)

if __name__ == '__main__':
    # test_data(NumPic_path,trainingSet_Size,validitySet_Size,testSet_Size)

    '''Main Training procedure'''
    allNumPicData = read_data(NumPic_path,trainingSet_Size,validitySet_Size,testSet_Size)
    currNeuralNetwork = NeuralNetwork(NeuronNum_list,allNumPicData)
    currNeuralNetwork.trainNeuralNetwork(batch_size,learning_rate,EpochNum)

    # currNeuralNetwork.check_gradient()
