import os
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
import pandas as pd
import numpy as np
from keras.utils import to_categorical   # 类别变量 转换为 one-hot变量

'''把样本集分为训练集（训练的时候再将这里分出来的训练集分为测试集和验证集）和测试集'''


TEST_SIZE = 0.1
RANDOM_STATE = 2019
best_accuracy = 0.0
NUM_CLASSES = 18
SAMPLE_SET_FILE_PATH = ".\\format_dir\\SakuraBuilding\\sample_set.csv"

# data preprocessing
'''
把一个样本集分为一个训练集（训练的时候再将这里分出来的训练集分为测试集和验证集）和一个测试集
并把训练集和测试集分别保存到csv文件中
'''
def divideOneSampleSet(file):
    sample_data = pd.read_csv(file)
    print(sample_data.columns)
    X = sample_data.values[:, 2:]
    y = keras.utils.to_categorical(sample_data.reference_tag, NUM_CLASSES)
    # print(y)
    # y = sample_data.values[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print("X_train:{} , y_train:{}".format(X_train.shape, y_train.shape))
    print("X_train.type:{} , y_train.type:{}".format(type(X_train), type(y_train)))

    print("X_train: {}\n".format(X_train))
    print("y_train: {}\n".format(y_train))
    # print("y_train: /n")
    # for df in y_train:
    #     print(df)
    print("X_test:{} , y_test:{}".format(X_test.shape, y_test.shape))
    print("X_test: {}\n".format(X_test))
    print("y_test: {}\n".format(y_test))
    row = y_test.shape[0]
    mergeTagAndfeatures(y_test, X_test, sample_data.columns, "test_set.csv")
    mergeTagAndfeatures(y_train, X_train, sample_data.columns, "train_set.csv")



    # np.savetxt('trainset.csv', trainset_array)
    # np.savetxt('testset.csv', testset_array, delimiter = ',')
    # !!!!!!must reshape the  X_train, X_test, y_train, y_test data to 3 dimensions!!!!!!!!

    # X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    # X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    # y_train = y_train.reshape((y_train.shape[0], NUM_CLASSES))
    # y_test = y_test.reshape((y_test.shape[0], NUM_CLASSES))

    # print("reshaped X_train: {}\n".format(X_train))
    # print("reshaped y_train: {}\n".format(y_train))
    return X_train, y_train, X_test, y_test

def mergeTagAndfeatures(y_test,X_test,columns,file_name):
    # 由one-hot转换为普通np数组
    y_test_array = [np.argmax(one_hot) for one_hot in y_test]
    print("y_test_array: {}\n".format(y_test_array))
    y_test_array = np.array(y_test_array).reshape(-1,1)
    print("after .... y_test_array: {}\n".format(y_test_array))
    dir_tag = [[0 for i in range(1)] for row in range(y_test.shape[0])]
    print("dir_tag: {}\n".format(dir_tag))
    testset_array = np.concatenate([y_test_array, dir_tag, X_test], axis=1)
    df = pd.DataFrame(data=testset_array,columns=columns,index=None)
    print(df)
    print("test_set: {}\n".format(testset_array))
    df.to_csv(file_name, sep=',',index=None)


# divideOneSampleSet(SAMPLE_SET_FILE_PATH)

