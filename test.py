import dataset
import MLP_BP
import cfg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def testset_establish():
    dataset1 = dataset.dataset()
    test_data = dataset1.test_data
    test_label = dataset1.test_label

    label = np.zeros((test_label.shape[0],1))
    for ih in range(label.shape[0]):
        for iw in range(10):
            if test_label[ih][iw]==1:
                label[ih][0]=iw
                break

    return test_data,label

def load_hidden_weight(model,path):
    weight = pd.read_csv(path[0], sep=' ',header=None)
    model.hidden_layer.W = np.array(weight.values,dtype='float32')
    bias = pd.read_csv(path[1],sep=' ',header=None)
    # print(bias.shape)
    model.hidden_layer.b = np.array(bias.values,dtype='float32').reshape(1,-1)
    # print(model.hidden_layer.b)


def load_output_weight(model,path):
    weight = pd.read_csv(path[0], sep=' ', header=None)
    model.output_layer.W = np.array(weight.values, dtype='float32')
    bias = pd.read_csv(path[1], sep=' ', header=None)
    # print(bias.shape)
    model.output_layer.b = np.array(bias.values, dtype='float32').reshape(1, -1)
    # print(model.output_layer.b)

def finale_test():
    hidden_path = ['./second_train_model/hidden_weight_epoch90.txt','./second_train_model/hidden_bias_epoch90.txt']
    output_path = ['./second_train_model/output_weight_epoch90.txt','./second_train_model/output_bias_epoch90.txt']
    test_data,label = testset_establish()

    # 这里，activate_function需要自己输入，要与训练模型时保持一致
    # 有点像torch里面load那个state字典，需要先初始化个模型
    mlp = MLP_BP.MLP(input_dim=28*28,hidden_element_number=300,out_classes=10,
                     activate_function='tanh')

    load_hidden_weight(mlp,hidden_path)
    load_output_weight(mlp,output_path)

    result_one_hot = mlp.forward(test_data)
    result = np.argmax(result_one_hot, axis=1).reshape(-1, 1)
    right = 0
    for i in range(label.shape[0]):
        if result[i][0] == label[i][0]:
            right += 1

    # print(right)

    print("总的 accuracy: %f" % (right / label.shape[0]))

    result_one_hot = mlp.forward(test_data[:5000])
    result = np.argmax(result_one_hot,axis=1).reshape(-1,1)
    right = 0
    error = []
    label1 = label[:5000]
    for i in range(label1.shape[0]):
        if result[i][0] == label1[i][0]:
            right+=1
        else:
            error.append(i)

    # print(error[:10])


    # print(right)

    print("前5000张 accuracy: %f"%(right/label1.shape[0]))

    result_one_hot = mlp.forward(test_data[5000:])
    result = np.argmax(result_one_hot, axis=1).reshape(-1, 1)
    right = 0
    error = []
    label1 = label[5000:]
    for i in range(label1.shape[0]):
        if result[i][0] == label1[i][0]:
            right += 1
        else:
            error.append(i)

    # print(error[:10])

    # print(right)

    print("后5000张 accuracy: %f" % (right / label1.shape[0]))

def test():
    hidden_path = ['./second_train_model/hidden_weight_epoch90.txt', './second_train_model/hidden_bias_epoch90.txt']
    output_path = ['./second_train_model/output_weight_epoch90.txt', './second_train_model/output_bias_epoch90.txt']
    test_data, label = testset_establish()
    mlp = MLP_BP.MLP(input_dim=28 * 28, hidden_element_number=300, out_classes=10)
    load_hidden_weight(mlp, hidden_path)
    load_output_weight(mlp, output_path)

    result_one_hot = mlp.forward(test_data)
    result = np.argmax(result_one_hot, axis=1).reshape(-1, 1)

    error1 = [8, 61, 63, 66, 124, 149, 193, 195, 211, 233]
    error2 = [54, 67, 68, 78, 140, 143, 165, 176, 183, 210]

    for each in error1:
        img_data = test_data[each].reshape(28,28)
        img_predict = result[each][0]
        img_label = label[each][0]
        plt.figure()
        plt.imshow(img_data,cmap='gray')
        plt.text(24,24,str(int(img_label)),color=[1,1,1],fontsize=20)
        plt.text(1,24,str(img_predict),color=[1,1,1],fontsize=20)
        plt.show()

    for each in error2:
        img_data = test_data[5000+each].reshape(28,28)
        img_predict = result[5000+each][0]
        img_label = label[5000+each][0]
        plt.figure()
        plt.imshow(img_data,cmap='gray')
        plt.text(24,24,str(int(img_label)),color=[1,1,1],fontsize=20)
        plt.text(1,24,str(img_predict),color=[1,1,1],fontsize=20)
        plt.show()
if __name__ == "__main__":
    finale_test()
    test()
