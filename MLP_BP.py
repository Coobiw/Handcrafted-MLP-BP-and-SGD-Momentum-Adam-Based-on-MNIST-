import numpy as np
import cfg
import math

class MLP():
    def __init__(self,input_dim,hidden_element_number,out_classes,lr=cfg.lr,
                 activate_function:str='tanh'):
        self.hidden_layer = self.Hidden_layer(input_dim,hidden_element_number,lr = lr,
                                    activate_function=activate_function)
        self.output_layer = self.Output_layer(hidden_element_number,out_classes,lr = lr)

    def forward(self,input):
        return self.output_layer.forward(self.hidden_layer.forward(input))

    def backward(self,label):
        self.hidden_layer.backward(self.output_layer.backward(label))

    def Adam_backward(self,label):
        self.hidden_layer.Adam_backward(self.output_layer.Adam_backward(label))

    class Hidden_layer():
        def __init__(self,input_dim,output_dim,lr,activate_function:str='tanh'):
            # Weight和Bias初始化
            self.W = np.random.randn(input_dim,output_dim) * math.sqrt(2/(input_dim+output_dim))
            self.b = np.zeros((1,output_dim))
            self.lr = lr

            activate_function = activate_function.lower() # 大写全部变小写
            assert activate_function in ['tanh','sigmoid','relu'],\
                'please input tanh/sigmoid/relu'

            if activate_function == 'tanh':
                self.activate_function = self.tanh
                self.diff_af = self.tanh_diff

            elif activate_function == 'sigmoid':
                self.activate_function = self.sigmoid
                self.diff_af = self.sigmoid_diff

            elif activate_function == 'relu':
                self.activate_function = self.ReLU
                self.diff_af = self.ReLU_diff


            # 用于BP计算的变量初始化
            self.last_Wg = np.zeros(self.W.shape)
            self.last_bg = np.zeros(self.b.shape)

            self.Eg1 = np.zeros(self.W.shape)
            self.Eg2 = np.zeros(self.W.shape)

            self.Ebg1 = np.zeros(self.b.shape)
            self.Ebg2 = np.zeros(self.b.shape)

        def forward(self,input_data): # input_data shape: (batch_size,input_dim)
            self.input = input_data
            self.a = input_data @ self.W + self.b
            self.h = self.activate_function(self.a)
            return self.h

        def backward(self,input_gradient): # input_gradient come from the αL/αh
            # input_gradient shape: (batch_size,output_dim)
            a_gradient = input_gradient * self.diff_af(self.a)

            b_gradient = input_gradient.mean(axis = 0)
            W_gradient = self.input.reshape(-1,self.input.shape[1],1) @ input_gradient.reshape(-1,1,input_gradient.shape[1])
            W_gradient = W_gradient.mean(axis=0)

            lamda = 0.9

            self.b = self.b - self.lr * (b_gradient + lamda * self.last_bg)
            self.W =self.W - self.lr * (W_gradient + lamda * self.last_Wg)

            self.last_Wg = W_gradient + lamda * self.last_Wg
            self.last_bg = b_gradient + lamda * self.last_bg

            return a_gradient @ self.W.T

        def Adam_backward(self,input_gradient):
            a_gradient = input_gradient * self.diff_af(self.a)
            b_gradient = input_gradient.mean(axis=0)
            W_gradient = self.input.reshape(-1, self.input.shape[1], 1) @ input_gradient.reshape(-1, 1, input_gradient.shape[1])
            W_gradient = W_gradient.mean(axis=0)

            beta = (0.9,0.99)
            epsilon = 1e-20
            self.Eg1 = beta[0] * self.Eg1 + (1-beta[0]) * W_gradient
            self.Eg2 = beta[1] * self.Eg2 + (1-beta[1]) * np.power(W_gradient,2)

            self.Ebg1 = beta[0] * self.Ebg1 + (1 - beta[0]) * b_gradient
            self.Ebg2 = beta[1] * self.Ebg2 + (1 - beta[1]) * np.power(b_gradient, 2)

            Eg1_hat = self.Eg1/(1-beta[0])
            Eg2_hat = self.Eg2/(1 - beta[1])
            Ebg1_hat = self.Ebg1/(1 - beta[0])
            Ebg2_hat = self.Ebg2/(1 - beta[1])

            self.W -= self.lr * (Eg1_hat/np.sqrt(Eg2_hat+epsilon))
            self.b -= self.lr * (Ebg1_hat/np.sqrt(Ebg2_hat+epsilon))

            return a_gradient @ self.W.T

        def sigmoid(self,x):
            return 1/(1+np.exp(-1*x))

        def ReLU(self,x):
            y = x.copy()
            y[y<=0] = 0
            return y

        def tanh(self,x):
            return 2*self.sigmoid(2*x) - 1

        def sigmoid_diff(self,x):
            y = self.sigmoid(x)
            return x*(1-x)

        def tanh_diff(self,x):
            y = self.tanh(x)
            return 1 - np.power(y,2)

        def ReLU_diff(self,x):
            y = x.copy()
            y[y <= 0] = 0
            y[y > 0] = 1
            return y

    class Output_layer():
        def __init__(self, input_dim, output_dim, lr):
            self.W = np.random.randn(input_dim, output_dim) * math.sqrt(2/(input_dim+output_dim))
            self.b = np.zeros((1, output_dim))
            self.lr = lr

            self.last_Wg = np.zeros(self.W.shape)
            self.last_bg = np.zeros(self.b.shape)

            self.Eg1 = np.zeros(self.W.shape)
            self.Eg2 = np.zeros(self.W.shape)

            self.Ebg1 = np.zeros(self.b.shape)
            self.Ebg2 = np.zeros(self.b.shape)
        def softmax(self,x):
            return np.exp(x)/(np.sum(np.exp(x),axis=1).reshape(-1,1))

        def forward(self,input_data):
            self.input = input_data
            self.a = input_data @ self.W + self.b
            self.h = self.softmax(self.a)
            return self.h

        def backward(self,label):
            o_gradient = self.h - label

            b_gradient = o_gradient.mean(axis=0)
            W_gradient = self.input.reshape(-1,self.input.shape[1],1) @ o_gradient.reshape(-1,1,o_gradient.shape[1])
            W_gradient = W_gradient.mean(axis=0)

            lamda = 0.9

            self.b = self.b - self.lr * (b_gradient + lamda * self.last_bg)
            self.W = self.W - self.lr * (W_gradient + lamda * self.last_Wg)

            self.last_Wg = W_gradient + lamda * self.last_Wg
            self.last_bg = b_gradient + lamda * self.last_bg

            return o_gradient @ self.W.T

        def Adam_backward(self,label):
            input_gradient = self.h - label

            b_gradient = input_gradient.mean(axis=0)
            W_gradient = self.input.reshape(-1, self.input.shape[1], 1) @ input_gradient.reshape(-1, 1, input_gradient.shape[1])
            W_gradient = W_gradient.mean(axis=0)

            beta = (0.9,0.99)
            epsilon = 1e-20
            self.Eg1 = beta[0] * self.Eg1 + (1-beta[0]) * W_gradient
            self.Eg2 = beta[1] * self.Eg2 + (1-beta[1]) * np.power(W_gradient,2)

            self.Ebg1 = beta[0] * self.Ebg1 + (1 - beta[0]) * b_gradient
            self.Ebg2 = beta[1] * self.Ebg2 + (1 - beta[1]) * np.power(b_gradient, 2)

            Eg1_hat = self.Eg1 / (1 - beta[0])
            Eg2_hat = self.Eg2 / (1 - beta[1])
            Ebg1_hat = self.Ebg1 / (1 - beta[0])
            Ebg2_hat = self.Ebg2 / (1 - beta[1])

            self.W -= self.lr * (Eg1_hat / np.sqrt(Eg2_hat + epsilon))
            self.b -= self.lr * (Ebg1_hat / np.sqrt(Ebg2_hat + epsilon))

            return input_gradient @ self.W.T





if __name__ == "__main__":
    import dataset
    import cfg
    import tqdm
    dataset1 = dataset.dataset()
    mlp = MLP(input_dim=28*28,hidden_element_number=1000,out_classes=10,
              lr=cfg.lr,activate_function='sigmoid')

    W1 = mlp.hidden_layer.W

    W2 = mlp.output_layer.W
    epoch = 2
    for i in tqdm.tqdm(range(epoch)):
        mlp.forward(dataset1.train_data[:1000])
        mlp.backward(dataset1.train_label[:1000])

    W11 = mlp.hidden_layer.W

    W21 = mlp.output_layer.W

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(W1,cmap="YlGn")

    plt.figure()
    plt.imshow(W11,cmap="YlGn")

    delta_W1 = W11 - W1
    plt.figure()
    plt.imshow(delta_W1, cmap="YlGn")

    plt.figure()
    plt.imshow(delta_W1[:20,:20],cmap="YlGn")

    # plt.figure()
    # plt.imshow(W2, cmap="YlGn")
    #
    # plt.figure()
    # plt.imshow(W21, cmap="YlGn")

    plt.show()