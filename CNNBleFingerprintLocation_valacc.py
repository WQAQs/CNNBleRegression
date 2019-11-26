import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv1D, Dropout, MaxPooling1D
import plotly.graph_objs as go
import plotly
from plotly import tools
from plotly.offline import iplot
from keras.utils import plot_model


TEST_SIZE = 0.2
RANDOM_STATE = 2019
best_accuracy = 0.0
NUM_CLASSES = 7
TrainFile = ".\\sample_dataset\\dim_from_18points_7classes_mac\\trainset_18points_7classes.csv"
TestFile = ".\\sample_dataset\\dim_from_18points_7classes_mac\\7points_7classes_testset0_5s.csv"
BestModleFilePath = ".\\my_best_model\\dim_from_18points_7classes_mac\\trainset_18points_7classes_test_plot.h5"
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    
# data preprocessing
def load_dataset(file):
    train_data = pd.read_csv(file)
    X = train_data.values[:,2:]
    y = keras.utils.to_categorical(train_data.referencePoint_tag, NUM_CLASSES)
    # print(y)
    # y = train_data.values[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    # print("X_train:{} , y_train:{}".format(X_train.shape, y_train.shape))
    # print("X_train: {}\n".format(X_train))
    # print("y_train: {}\n".format(y_train))
    # print("y_train: /n")
    # for df in y_train:
    #     print(df)
    # print("X_test:{} , y_test:{}".format(X_test.shape, y_test.shape))
    # !!!!!!must reshape the  X_train, X_test, y_train, y_test data to 3 dimensions!!!!!!!!
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    y_train = y_train.reshape((y_train.shape[0], NUM_CLASSES))
    y_test = y_test.reshape((y_test.shape[0], NUM_CLASSES))
    # print("reshaped X_train: {}\n".format(X_train))
    # print("reshaped y_train: {}\n".format(y_train))
    return X_train, y_train, X_test, y_test

def evaluate_model(trainX, trainy, testX, testy,i):
    global best_accuracy
    verbose, epochs, batch_size = 0, 10, 32
    # n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
#     model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))  (x_train.shape[1],1)
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(trainX.shape[1], 1)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(trainy.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    # model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=1)
    history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=1,validation_data=(testX, testy)).history
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    plot_model(model, to_file='model.png')
    plotfile_name = "train_model_" + str(i) + "accuracy-loss"
    plot_accuracy_and_loss(model,plotfile_name,history,i)
    if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save(BestModleFilePath)
    return accuracy

def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores) # np.mean(): 求平均值   np.std()：求标准差
    print('summarize Accuracy: %.3f%% (+/-%.3f)' % (m, s))


def run_experiment(repeats=1):
    # load train data
    trainX, trainy, testX, testy = load_dataset(TrainFile)
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy,r)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)
    print('best_score:%.3f \n' % (best_accuracy * 100))

def create_trace(x, y, ylabel, color):
    trace = go.Scatter(
        x=x, y=y,
        name=ylabel,
        marker=dict(color=color),
        mode="markers+lines",
        text=x
    )
    return trace


def plot_accuracy_and_loss(train_model,plotfile_name,history,i):
    # hist = train_model.history
    hist = history
    acc = hist['acc']
    val_acc = hist['val_acc']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = list(range(1, len(acc) + 1))

    trace_ta = create_trace(epochs, acc, "Training accuracy", "Green")
    trace_va = create_trace(epochs, val_acc, "Validation accuracy", "Red")
    trace_tl = create_trace(epochs, loss, "Training loss", "Blue")
    trace_vl = create_trace(epochs, val_loss, "Validation loss", "Magenta")

    fig = plotly.subplots.make_subplots(rows=1, cols=2, subplot_titles=('Training and validation accuracy',
                                                              'Training and validation loss'))
    fig = plotly.subplots.make_subplots(rows=1,cols=2, subplot_titles=('Training and validation accuracy',
                                                                 'Training and validation loss'))
    fig.append_trace(trace_ta, 1, 1)
    fig.append_trace(trace_va, 1, 1)
    fig.append_trace(trace_tl, 1, 2)
    fig.append_trace(trace_vl, 1, 2)
    fig['layout']['xaxis'].update(title='Epoch' + str(i + 1))
    fig['layout']['xaxis2'].update(title='Epoch' + str(i + 1))
    fig['layout']['yaxis'].update(title='Accuracy', range=[0, 1])
    fig['layout']['yaxis2'].update(title='Loss', range=[0, 15])

    iplot(fig, filename=plotfile_name)


# plot_accuracy_and_loss(train_model)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    # run the experiment
    run_experiment()

    # load the best model and test
    trainX, trainy, testX, testy = load_dataset(TestFile)
    my_model = keras . models . load_model(BestModleFilePath)
    my_model.summary()
    # plot_accuracy_and_loss()
    preds = my_model . evaluate( testX, testy )
    print("my best model = trainset_18points_7classes.h5")
    print("test file = {}".format(TestFile))
    print("test result:")
    print ( "my_best_model Loss = " + str( preds[0] ) )
    print ( "my_best_model Test Accuracy = " + str( preds[1] ) )
