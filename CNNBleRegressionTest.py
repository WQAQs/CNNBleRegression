import numpy as np
import math
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv1D, Dropout, MaxPooling1D
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import globalConfig


TEST_SIZE = 0.1
RANDOM_STATE = 2019
best_accuracy = 0.0
NUM_CLASSES = 18
EPOCHS = 1000
SampleFile = '.\\sample_set.csv'
TrainFile = '.\\train_set.csv'
PreModleFile = "my_model1.h5"
CurrentModleFile = "current_model1.h5"

TestFile = '.\\test_set19_3s.csv'

Error_Distance_File = '.\\error_distance_over1m_19_3s.csv'


def divide_sample_dataset(sample_dataset_file):
    sample_dataset = pd.read_csv(sample_dataset_file)
    train_dataset = sample_dataset.sample(frac=0.8,random_state=0)
    test_dataset = sample_dataset.drop(train_dataset.index)
    return train_dataset, test_dataset

def load_dataset(dataset):
    reference_tag = dataset.values[:, 0]
    data_input = dataset.values[:,5:] #包括index=5
    data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
    coordinates = dataset.values[:,1:3] #包括index=1，不包括index=3
    return data_input, coordinates,reference_tag

def load_data(data_file):
    dataset = pd.read_csv(data_file)
    data_input , coordinates, reference_tag = load_dataset(dataset)
    return data_input,coordinates,reference_tag

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

# 通过为每个完成的时期打印一个点来显示训练进度
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

def retrain_model(train_data_input, train_coordinates):

    # load the pre trained model
    pre_model = keras.models.load_model(PreModleFile)
    pre_model.summary()
    model = pre_model

    # 使用原始方式fit network
    history = model.fit(train_data_input,train_coordinates,epochs=EPOCHS,validation_split=0.2,verbose=1,callbacks=[PrintDot()])

    # 使用early stop方法来fit network
    # early stop方法：如果经过一定数量的epochs后没有改进，则自动停止训练
    # patience 值用来检查改进 epochs 的数量
    # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    # history = model.fit(train_data_input, train_coordinates, epochs=EPOCHS,
    #                     validation_split=0.2, verbose=1, callbacks=[early_stop, PrintDot()])


    # 可视化训练进度
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    plot_history(history)

    # 测试集测试
    loss, mae, mse = model.evaluate(test_data_input, test_coordinates, verbose=2)
    print("Testing set Mean Abs Error: {:5.2f} ".format(mae))

    # save the model
    model.save(CurrentModleFile)


def train_model(train_data_input, train_coordinates):
    print(train_coordinates.shape[1])
    model = Sequential()

    '''Choose a model to train'''

    # '''1. CNN model'''
    # model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(train_data_input.shape[1], 1)))
    # model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Flatten())
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(2, activation='relu'))

    '''2. MLP model'''
    n = 20
    model.add(Dense(512,input_shape=(train_data_input.shape[1],),activation='relu'))
    for i in range(n):
        model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='relu'))

    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])


    # fit network
    # history = model.fit(train_data_input,train_coordinates,epochs=EPOCHS,validation_split=0.2,verbose=1,callbacks=[PrintDot()])

    # 使用early stop方法来fit network
    # early stop方法：如果经过一定数量的epochs后没有改进，则自动停止训练
    # patience 值用来检查改进 epochs 的数量
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(train_data_input, train_coordinates, epochs=EPOCHS,
                        validation_split=0.2, verbose=1, callbacks=[early_stop, PrintDot()])
    # 可视化训练进度
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    plot_history(history)

    # 测试集测试
    loss, mae, mse = model.evaluate(test_data_input, test_coordinates, verbose=2)
    print("Testing set Mean Abs Error: {:5.2f} ".format(mae))

    # save the model
    model.save(CurrentModleFile)


def change_to_right(one_hot_labels):
    right_labels=[]
    for x in one_hot_labels:
        for i in range(0,NUM_CLASSES):
            if x[i]==1:
                # print("label_real:{}".format(i + 1))
                right_labels.append(i)
    return right_labels

def calculate_distance(result):
    pred_coordinatesx,pred_coordinatesy = result[0], result[1]
    true_coordinatesx,true_coordinatesy = result[2], result[3]
    error_x2, error_y2 = math.pow(pred_coordinatesx - true_coordinatesx, 2), \
                         math.pow(pred_coordinatesy - true_coordinatesy, 2)
    error_distance = math.sqrt(error_x2 + error_y2)  # 求平方根
    return error_distance



config = tf.ConfigProto()
config.gpu_options.allow_growth = True


'''load train data and test data'''
# train_dataset, test_dataset = divide_sample_dataset(SampleFile)
# train_data_input, train_coordinates, train_reference_tag = load_dataset(train_dataset)
# test_data_input, test_coordinates, test_reference_tag = load_dataset(test_dataset)

'''only load test data'''
test_data_input, test_coordinates,test_reference_tag = load_data(TestFile)

'''train/retrain the model'''
# train and save the model
# retrain_model(train_data_input, train_coordinates, my_model)
# train_model(train_data_input,train_coordinates)

'''test the current model'''
current_model = keras.models.load_model(CurrentModleFile)

'''evaluate the model'''
# get the prediction results
pred_results = current_model.predict(test_data_input)
true_coordinatesx, true_coordinatesy = test_coordinates[:, 0], test_coordinates[:, 1]
pred_coordinatesx, pred_coordinatesy= pred_results[:, 0], pred_results[:, 1]

# evaluate the model's Loss
pred_evaluate = current_model.evaluate(test_data_input, test_coordinates, verbose=1)
print("my_best_model Loss = " + str(pred_evaluate[0]))

# plot prediction and true data
plt.figure()
plt.scatter(pred_coordinatesx, pred_coordinatesy,c='blue')
plt.scatter(true_coordinatesx,true_coordinatesy,c='red')
plt.xlabel('coordinate_x(/m)')
plt.ylabel('coordinate_y(/m)')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
# plt.show()  # 一定要加这个，否则在pycharm中显示不出来绘图的窗口

# show Error distribution
results = np.hstack((pred_results,test_coordinates))
error_distance = list(map(calculate_distance,results))
error_distance_bypoint = np.hstack((np.array(error_distance).reshape(-1, 1), test_reference_tag.reshape(-1, 1)))
error_distance_bypoint_df = pd.DataFrame(error_distance_bypoint,columns=['error_distance','test_reference_tag'])
# error_over1m_df = pd.DataFrame(error_distance_bypoint_df['error_distance']>1.0)
error_over1m_df = error_distance_bypoint_df[error_distance_bypoint_df['error_distance']>1.0]
error_over1m_df = error_over1m_df.sort_values(by='error_distance')
error_mean = error_over1m_df['error_distance'].mean()
error_over1m_df.to_csv(Error_Distance_File, index=False, encoding='utf-8')
len = len(error_distance)
plt.figure()
plt.scatter(list(range(0,len)),error_distance)
# plt.hist(error_distance, bins=100)
plt.show()
plt.xlabel("Prediction Distance Error(/m)")







