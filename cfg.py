train_data_path = "./MNIST_dataset/train-images.idx3-ubyte"
train_label_path = "./MNIST_dataset/train-labels.idx1-ubyte"

test_data_path = "./MNIST_dataset/t10k-images.idx3-ubyte"
test_label_path = "./MNIST_dataset/t10k-labels.idx1-ubyte"

# default of the hyper para
epoch = 50
lr = 1e-2
batch_size = 256


if __name__ == "__main__":
    import struct
    import numpy as np
    data = open(train_data_path,"rb").read()  #read the data from the format file
    # print(data)
    offset = 0
    fmt_header = '>iiii'
    magic_number,img_num,row,column = struct.unpack_from(fmt_header,data,offset)
    print("魔数:{}\t img_num:{}\t row:{}\t column:{}\t".format(magic_number,img_num,row,column))

    # method 1:
    # img1 = np.zeros((row,column),dtype='uint8')
    # offset1 = 16
    # for ih in range(row):
    #     for iw in range(column):
    #         img1[ih][iw] = int(struct.unpack_from('>B',data,offset1)[0])
    #         offset1 +=1

    # method 2:
    # offset += struct.calcsize(fmt_header)
    # print(offset)
    # img1_flatten = struct.unpack_from('>'+str(row*column)+'B',data,offset)
    # img1 = np.array(img1_flatten,dtype = 'uint8').reshape(row,column)

    # 全读取method
    offset += struct.calcsize(fmt_header)
    print(offset)
    fmt_img_header = '>' + str(img_num*row*column) + 'B'
    img_flatten = struct.unpack_from(fmt_img_header,data,offset)

    img_dataset = np.array(img_flatten,dtype='uint8').reshape(img_num,row,column)
    offset += struct.calcsize(fmt_img_header)
    print(offset)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(img_dataset[img_num-1],cmap="gray")
    plt.show()
