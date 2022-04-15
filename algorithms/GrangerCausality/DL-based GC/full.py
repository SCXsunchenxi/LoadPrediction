import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square
import gc
import os
import random
from copy import deepcopy
from lstm_model import LSTM
import logging


logging.basicConfig(format="[%(asctime)s] [%(filename)s] %(message)s",
                    level=logging.INFO, filename='./full.log',
                    )

os.environ["CUDA_VISIBLE_DEVICES"] = "9"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


seq_length = 10
target_variable = 0


batch_size = 16
num_epochs = 30000
origin_learning_rate = 0.001
hidden_size = 700
num_layers = 1
mlp_hidden_size = 128
weight_decay = 1e-2
patience = 100
threshold = 0.02
is_random = 'ordered'
print(
    "target:%d,%s, seq_length: %d, batch_size: %d, origin_learning_rate: %f, hidden_size: %d, num_layers: %d, weight_decay: %f, patience: %d, threshold: %f" % (
    target_variable,is_random, seq_length, batch_size, origin_learning_rate, hidden_size, num_layers, weight_decay, patience,
    threshold))
logging.info(
    'target:{},{}, seq_length: {}, batch_size: {}, origin_learning_rate: {}, hidden_size: {}, num_layers: {}, weight_decay: {}, patience: {}, threshold: {}'.format(
        target_variable,is_random, seq_length, batch_size, origin_learning_rate, hidden_size, num_layers, weight_decay, patience,
        threshold
    ))




raw_data=pd.read_csv('./lorenz96_data.csv',header=None).values
print(raw_data.shape)


# 以滑动窗口的形式设置数据集,处理含间隔的连续数据
def sliding_windows(data):
    i=0
    x=[]
    y=[]
    while i+seq_length<data.shape[0]:
        past_x=data[i:(i+seq_length)]
        future_y=data[i+seq_length]
        x.append(past_x)
        y.append(future_y)
        i=i+1
    return np.array(x),np.array(y)


merge_x, merge_y = sliding_windows(raw_data)
print("merge_x ", merge_x.shape)  # merge_x  (990, 10, 10)
print("merge_y", merge_y.shape)  # merge_y (990, 10)

# 训练数据,转成tensor
train_data_x = Variable(torch.Tensor(np.array(merge_x)))
# print(train_data_x.shape)
train_data_y = Variable(torch.Tensor(np.array(merge_y)))
train_data_x = train_data_x.to(device)
train_data_y = train_data_y.to(device)


def get_mini_batch_random(X, Y, batch_size):
    mini_batches = []
    m = X.shape[0]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :]
    shuffled_Y = Y[permutation, :]
    for i in range(batch_size, m, batch_size):
        mini_batch_x = shuffled_X[i:i + batch_size]
        mini_batch_y = shuffled_Y[i:i + batch_size]
        mini_batches.append((mini_batch_x, mini_batch_y))
    mini_batches.append((shuffled_X[0:batch_size], shuffled_Y[0:batch_size]))
    return mini_batches


def get_mini_batch(X, Y, batch_size):
    mini_batches = []
    m = X.shape[0]
    for i in range(0, m, batch_size):
        mini_batch_x = X[i:i + batch_size]
        mini_batch_y = Y[i:i + batch_size]
        mini_batches.append((mini_batch_x, mini_batch_y))
    return mini_batches


minibatches = get_mini_batch(merge_x, merge_y, batch_size)
print(len(minibatches))




# Training
input_size = merge_y.shape[1]
num_classes = merge_y.shape[1]

num_of_epoch = 0
max_of_epoch = 1
residual_variance_list = []
mean_squared_error_list = []
mean_absolute_error_list = []
r2_score_list = []

judge_point = []

for i in range(0,200):
    judge_point.append(50 * i)


def compute_prediction_error(lstm):

    predict_output = lstm(train_data_x)  #
    predict_output = predict_output.data.cpu().numpy()  # torch.Size([8924, 2068])

    # data_predict = predict_output[:, target_variable].reshape(-1,1)  # [990,1]
    # true_y = merge_y[:, target_variable].reshape(-1,1)  # (8924, 188)

    data_predict=predict_output
    true_y=merge_y

    error_list = []
    # MSE
    squared_error = mean_squared_error(true_y, data_predict)
    print("MSE: ", squared_error)
    logging.info('MSE: {:.8f}'.format(squared_error))
    # MAE
    absolute_error = mean_absolute_error(true_y, data_predict)
    print("MAE: ", absolute_error)
    logging.info('MAE: {:.8f}'.format(absolute_error))
    # residual variance
    for i in range(data_predict.shape[0]):
        for j in range(data_predict.shape[1]):
            error_list.append(true_y[i, j] - data_predict[i, j])
    residual_variance = np.var(error_list)
    print("残差的方差:", residual_variance)
    logging.info('残差的方差: {:.8f}'.format(residual_variance))

    residual_variance_list.append(residual_variance)
    mean_squared_error_list.append(squared_error)
    mean_absolute_error_list.append(absolute_error)



while num_of_epoch < max_of_epoch:
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers, device)
    # lstm = LSTM_MLP(num_classes, input_size, hidden_size, mlp_hidden_size, num_layers,device)
    lstm = lstm.to(device)

    best_model = None
    best_loss = np.inf
    best_interation = None

    criterion = torch.nn.MSELoss()  # mean-squared error for regression, L2范数的平方
    criterion_test = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(lstm.parameters(), lr=origin_learning_rate, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience,threshold=threshold, verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    # optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

    # Train the model
    train_hist = []
    test_hist = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        # minibatches = get_mini_batch_random(merge_x, merge_y, batch_size)

        for minibatch in minibatches:
            (batch_x, batch_y) = minibatch
            batch_x = Variable(torch.Tensor(np.array(batch_x))).to(device)
            batch_y = Variable(torch.Tensor(np.array(batch_y))).to(device)
            outputs = lstm(batch_x)
            optimizer.zero_grad()
            loss = criterion(outputs, batch_y)+0*torch.sum(torch.abs(lstm.lstm.weight_ih_l0))
            train_hist.append(loss.item())
            loss.backward()
            optimizer.step()

        print("Epoch: %d, train loss: %f" % (epoch, train_hist[-1]))
        logging.info('Epoch: {},  train loss: {:.8f}'.format(epoch, train_hist[-1]))
        temp_lr = optimizer.state_dict()['param_groups'][0]['lr']
        # scheduler.step(train_hist[-1])
        scheduler.step()
        if optimizer.state_dict()['param_groups'][0]['lr'] != temp_lr:
            logging.info('Epoch: {} reducing learning rate of group 0 to {}'.format(epoch, optimizer.state_dict()[
                'param_groups'][0]['lr']))
        # if (epoch + 1) in judge_point:
        #     logging.info(
        #         '{} seq_length: {}, batch_size: {}, origin_learning_rate: {}, hidden_size: {}, num_layers: {}, weight_decay: {}, patience: {}, threshold: {}'.format(
        #             is_random, seq_length, batch_size, origin_learning_rate, hidden_size, num_layers,
        #             weight_decay, patience, threshold
        #         ))
        #     compute_prediction_error(lstm)

        compute_prediction_error(lstm)

        if len(residual_variance_list) > 0:
            if residual_variance_list[-1] < best_loss:
                best_loss = residual_variance_list[-1]
                best_model = deepcopy(lstm)
                best_interation = epoch
            elif (epoch - best_interation) == 250:
                print("stop training")
                # logging.info('stop training')
                break

        if epoch > 300:
            loss_list = residual_variance_list[-5:]
            # loss_list = train_hist[-200:]
            if (np.mean(loss_list) - best_loss) / best_loss < 0.01:
                print("stop training")
                # logging.info('stop training')
                break

        if epoch + 1 == 600:
            break


    # if os.path.exists('./full.pkl'):
    #     last_best_model = torch.load('./full.pkl')
    #     compute_prediction_error(last_best_model)  # 倒数第二个loss是上一次的loss
    #     compute_prediction_error(best_model)
    #     if residual_variance_list[-1] < residual_variance_list[-2]:
    #         torch.save(best_model, './full.pkl')
    # else:
    torch.save(best_model, './model_trained/without_weight_decay/hidden'+str(hidden_size)+'full.pkl')

    # Testing
    lstm.eval()  # 进入测试模式
    print("进入测试")

    compute_prediction_error(best_model)
    # compute_loss(lstm)

    num_of_epoch = num_of_epoch + 1
    del lstm
    gc.collect()


