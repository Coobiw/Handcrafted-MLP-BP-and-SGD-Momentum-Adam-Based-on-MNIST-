import dataset
import MLP_BP
import cfg
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import threading
import tqdm
import argparse
import os

def dataset_establish(val_ratio,seed):
    dataset1 = dataset.dataset()
    finale_train_data = dataset1.train_data
    finale_train_label = dataset1.train_label

    test_data = dataset1.test_data
    test_label = dataset1.test_label

    train_data,train_label,val_data,val_label = random_split(finale_train_data,finale_train_label,val_ratio,seed)

    return finale_train_data,finale_train_label,train_data,train_label,val_data,val_label,test_data,test_label

def random_split(data_X,data_Y,val_ratio,seed):
    random.seed(seed)
    m = data_X.shape[0]
    val_num = int(val_ratio * m)
    train_num = m - val_num
    total_list = [i for i in range(m)]
    train_index = random.sample(total_list, train_num)
    train_index.sort()
    for each in train_index:
        total_list.remove(each)
    train_X = np.zeros((train_num, data_X.shape[1]))
    train_Y = np.zeros((train_num, 10))
    val_X = np.zeros((val_num, data_X.shape[1]))
    val_Y = np.zeros((val_num, 10))

    # print(train_index)
    for i, each in enumerate(train_index):
        train_X[i] = data_X[each]
        train_Y[i] = data_Y[each]

    for i, each in enumerate(total_list):
        val_X[i] = data_X[each]
        val_Y[i] = data_Y[each]

    return train_X,train_Y,val_X,val_Y

# 划分batch
def split_batch(index_list,batch_size):
    batch_index = random.sample(index_list, batch_size)
    for each in batch_index:
        index_list.remove(each)

    return index_list,np.array(batch_index)

# cross entropy loss without softmax
def loss_function(y_predict,y):
    total_loss = np.sum(y * np.log(y_predict),axis=1)
    return -1*np.mean(total_loss,axis=0)

# save the model
def model_save(mlp,save_dir:str,epoch_num:int):
    try:
        os.chdir(save_dir)
    except FileNotFoundError:
        os.mkdir(save_dir)
        os.chdir(save_dir)
    finally:
        print('start saving...')

    f1 = open('hidden_weight_epoch'+str(epoch_num)+'.txt', 'w')
    f1.close()
    f2 = open('hidden_bias_epoch'+str(epoch_num)+'.txt','w')
    f2.close()
    f3 = open('output_weight_epoch'+str(epoch_num)+'.txt', 'w')
    f3.close()
    f4 = open('output_bias_epoch'+str(epoch_num)+'.txt','w')
    f4.close()

    with open("hidden_weight_epoch"+str(epoch_num)+'.txt', "a") as f:
        for ih in range(mlp.hidden_layer.W.shape[0]):
            for iw in range(mlp.hidden_layer.W.shape[1]):
                f.write(str(mlp.hidden_layer.W[ih][iw]))
                if iw != mlp.hidden_layer.W.shape[1] - 1:
                    f.write(' ')
            f.write('\n')

    with open("output_weight_epoch"+str(epoch_num)+'.txt', "a") as f:
        for ih in range(mlp.output_layer.W.shape[0]):
            for iw in range(mlp.output_layer.W.shape[1]):
                f.write(str(mlp.output_layer.W[ih][iw]))
                if iw != mlp.output_layer.W.shape[1] - 1:
                    f.write(' ')
            f.write('\n')

    with open("hidden_bias_epoch"+str(epoch_num)+'.txt', "a") as f:
        for iw in range(mlp.hidden_layer.b.shape[1]):
            f.write(str(mlp.hidden_layer.b[0][iw]))
            if iw != mlp.hidden_layer.b.shape[1] - 1:
                f.write(' ')

    with open("output_bias_epoch"+str(epoch_num)+'.txt', "a") as f:
        for iw in range(mlp.output_layer.b.shape[1]):
            f.write(str(mlp.output_layer.b[0][iw]))
            if iw != mlp.output_layer.b.shape[1] - 1:
                f.write(' ')



def finale_train(optimizer = 'SGD',lr = cfg.lr,batch_size=cfg.batch_size,
    epoch = cfg.epoch,save_dir:str='.',activate_function:str='tanh',
    plot_flag:bool=False,save_flag:bool=True):
    # 划分训练集、验证集（事实上MNIST不用分）
    finale_train_data, finale_train_label, train_data, train_label, val_data, val_label, \
    test_data, test_label = dataset_establish(1 / 6, 608)

    mlp = MLP_BP.MLP(input_dim=28 * 28, hidden_element_number=300, out_classes=10,lr=lr,
                     activate_function=activate_function)

    # 记录最初的W和b
    W1i = mlp.hidden_layer.W
    W2i = mlp.output_layer.W
    b1i = mlp.hidden_layer.b
    b2i = mlp.output_layer.b

    tloss_list = []
    taccuracy_list = []
    vloss_list = []
    vaccuracy_list = []

    # label 转换为one-hot 编码
    test_result = np.zeros((test_label.shape[0], 1))
    for ih in range(test_result.shape[0]):
        for iw in range(10):
            if test_label[ih][iw] == 1:
                test_result[ih][0] = iw
                break

    train_result = np.zeros((finale_train_label.shape[0], 1))
    for ih in range(train_result.shape[0]):
        for iw in range(10):
            if finale_train_label[ih][iw] == 1:
                train_result[ih][0] = iw
                break

    for i in tqdm.tqdm(range(epoch)):
        print("%d epoch" % (i + 1), end="\n")
        index_list = [i for i in range(finale_train_data.shape[0])]
        turn = 1
        tloss = 0.
        vloss = 0.

        while (len(index_list) >= batch_size):
            # print("%d turn" % turn, end="\t")
            index_list, batch_index = split_batch(index_list, batch_size)
            loss = loss_function(mlp.forward(finale_train_data[batch_index]), finale_train_label[batch_index])
            if optimizer == 'Adam':
                mlp.Adam_backward(finale_train_label[batch_index])
            elif optimizer == 'SGD':
                mlp.backward(finale_train_label[batch_index])
            # print("train_loss: %.5f"%loss,end="\n")
            tloss += loss
            turn += 1

        if len(index_list) != 0:
            # print("%d turn" % turn, end="\t")
            turn += 1
            batch_index = index_list
            loss = loss_function(mlp.forward(finale_train_data[batch_index]),
                                 finale_train_label[batch_index])

            if optimizer == 'Adam':
                mlp.Adam_backward(finale_train_label[batch_index])
            elif optimizer == 'SGD':
                mlp.backward(finale_train_label[batch_index])

            tloss += loss
            print("train_loss: %.5f" % (tloss / turn), end="\t")
            tloss_list.append(tloss / turn)
            result_one_hot = mlp.forward(finale_train_data)
            result = np.argmax(result_one_hot, axis=1).reshape(-1, 1)
            right = 0
            for i in range(train_result.shape[0]):
                if result[i][0] == train_result[i][0]:
                    right += 1

            print("train_accuracy: %f" % (right / train_result.shape[0]),end='\t')
            taccuracy_list.append(right/train_result.shape[0])
            val_loss = loss_function(mlp.forward(test_data), test_label)
            print("val_loss: %.5f" % val_loss,end='\t')
            vloss_list.append(val_loss)

            result_one_hot = mlp.forward(test_data)
            result = np.argmax(result_one_hot, axis=1).reshape(-1, 1)
            right = 0
            for i in range(test_result.shape[0]):
                if result[i][0] == test_result[i][0]:
                    right += 1

            print("test_accuracy: %f" % (right / test_result.shape[0]),end='\n')
            vaccuracy_list.append(right/test_result.shape[0])
        else:
            print("train_loss: %.5f" % (tloss / turn), end="\t")
            tloss_list.append(tloss / turn)
            result_one_hot = mlp.forward(finale_train_data)
            result = np.argmax(result_one_hot, axis=1).reshape(-1, 1)
            right = 0
            for i in range(train_result.shape[0]):
                if result[i][0] == train_result[i][0]:
                    right += 1

            print("train_accuracy: %f" % (right / train_result.shape[0]), end='\t')
            taccuracy_list.append(right / train_result.shape[0])
            val_loss = loss_function(mlp.forward(test_data), test_label)
            print("val_loss: %.5f" % val_loss,end='\t')
            vloss_list.append(val_loss)

            result_one_hot = mlp.forward(test_data)
            result = np.argmax(result_one_hot, axis=1).reshape(-1, 1)
            right = 0
            for i in range(test_result.shape[0]):
                if result[i][0] == test_result[i][0]:
                    right += 1

            print("test_accuracy: %f" % (right / test_result.shape[0]), end='\n')
            vaccuracy_list.append(right / test_result.shape[0])

    print("finish training...")

    if save_flag==True:
        model_save(mlp=mlp,save_dir=save_dir,epoch_num=epoch)

    # 展示一些 W，b的图片
    if plot_flag == True:
        x_plot = np.arange(epoch) + 1
        plt.figure()
        plt.plot(x_plot, np.array(tloss_list),color='green',label='train_loss')
        plt.plot(x_plot,np.array(vloss_list),color='pink',label='val_loss')
        plt.legend(loc='best')

        plt.figure()
        plt.plot(x_plot,np.array(taccuracy_list),color='green',label='train_accuracy')
        plt.plot(x_plot,np.array(vaccuracy_list),color='pink',label='val_accuracy')
        plt.legend(loc='best')

        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(W1i, cmap='YlGn')
        plt.subplot(1, 2, 2)
        plt.imshow(mlp.hidden_layer.W, cmap='YlGn')

        delta_W1 = mlp.hidden_layer.W - W1i
        plt.figure()
        plt.imshow((delta_W1 / mlp.hidden_layer.W), cmap='YlGn')

        plt.figure()
        plt.imshow((delta_W1/mlp.hidden_layer.W)[:20, :20])

        delta_W2 = mlp.output_layer.W - W2i
        plt.figure()
        plt.imshow((delta_W2 / mlp.output_layer.W), cmap='YlGn')

        plt.figure()
        plt.imshow((delta_W2 / mlp.output_layer.W)[:20, :20])

        plt.figure()
        plt.imshow(mlp.hidden_layer.b-b1i,cmap='YlGn')

        plt.figure()
        plt.imshow(((mlp.hidden_layer.b-b1i)[0][:20]).reshape(1,20),cmap='YlGn')

        plt.figure()
        plt.imshow(mlp.output_layer.b - b2i, cmap='YlGn')
        plt.show()

def arg_parser():
    parser = argparse.ArgumentParser(description='hyper para of trainer')
    parser.add_argument('--lr',type=float,default=cfg.lr,help='lr of the trainer')
    parser.add_argument('--batch-size',type=int,default=cfg.batch_size)
    parser.add_argument('--optimizer',type=str,default='SGD',
                        choices=['SGD','Adam'],help='please choose SGD/Adam')
    parser.add_argument('--epoch',type=int,default=cfg.epoch)
    parser.add_argument('--activate-function',type=str,default='tanh',
                choices=['relu','sigmoid','tanh'],help='choose tanh/sigmoid/relu')
    parser.add_argument('--save-dir',type=str,help='the save directory of the trained model')

    return parser




if __name__ == "__main__":
    parser = arg_parser().parse_args()
    finale_train(optimizer=parser.optimizer,lr = parser.lr,epoch=parser.epoch,
                 batch_size=parser.batch_size,
                 save_dir=parser.save_dir,activate_function=parser.activate_function,
                 plot_flag=False,save_flag=True)