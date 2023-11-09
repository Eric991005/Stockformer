import logging
import numpy as np
import pandas as pd
import os
import pickle
import sys
import torch
import math
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from statsmodels.tsa.seasonal import STL

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

# metric
def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        wape = np.divide(np.sum(mae), np.sum(label))
        wape = np.nan_to_num(wape * mask)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)

        # sort the predicted and true values in descending order
        sort_idx = np.argsort(-pred, axis=-1)
        pred_sorted = np.take_along_axis(pred, sort_idx, axis=-1)
        label_sorted = np.take_along_axis(label, sort_idx, axis=-1)
        
        # calculate the number of rows that represent 5% of the total
        num_rows = pred_sorted.shape[-1]
        num_top_rows = int(num_rows * 0.05)
        
        # compare the signs of the top rows and calculate the accuracy
        top_pred = pred_sorted[..., :num_top_rows]
        top_label = label_sorted[..., :num_top_rows]
        top_pred = pred_sorted
        top_label = label_sorted
        top_correct = np.sign(top_pred) == np.sign(top_label)
        accuracy = np.mean(top_correct)
        
    return mae, rmse, mape, accuracy



def _compute_loss(y_true, y_predicted):
    return masked_mae(y_predicted, y_true, 0.0)


# def masked_smape(preds, labels, null_val=np.nan):
#     if np.isnan(null_val):
#         mask = ~torch.isnan(labels)
#     else:
#         mask = (labels != null_val)
#     mask = mask.float()
#     mask /= torch.mean(mask)
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

#     numerator = torch.abs(preds - labels)
#     denominator = (torch.abs(preds) + torch.abs(labels)) / 2.0
#     smape = 200.0 * torch.mean((numerator / denominator) * mask)
    
#     return smape


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def seq2instance(data, P, Q):
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, dims))
    y = np.zeros(shape = (num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

def bonus_seq2instance(data, P, Q):
    num_step, dims, N = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, dims, N))
    y = np.zeros(shape = (num_sample, Q, dims, N))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

def disentangle(data, w, j):
    # Disentangle
    dwt = DWT1DForward(wave=w, J=j)
    idwt = DWT1DInverse(wave=w)
    torch_traffic = torch.from_numpy(data).transpose(1,-1).reshape(data.shape[0]*data.shape[2], -1).unsqueeze(1)
    torch_trafficl, torch_traffich = dwt(torch_traffic.float())
    placeholderh = torch.zeros(torch_trafficl.shape)
    placeholderl = []
    for i in range(j):
        placeholderl.append(torch.zeros(torch_traffich[i].shape))
    torch_trafficl = idwt((torch_trafficl, placeholderl)).reshape(data.shape[0],data.shape[2],1,-1).squeeze(2).transpose(1,2)
    torch_traffich = idwt((placeholderh, torch_traffich)).reshape(data.shape[0],data.shape[2],1,-1).squeeze(2).transpose(1,2)
    trafficl = torch_trafficl.numpy()
    traffich = torch_traffich.numpy()
    return trafficl, traffich


def stl_decomposition(series):
    """
    对ndarray的每一列进行STL分解，返回趋势项和季节项
    
    Args:
    series: ndarray, shape (n_samples, n_features)
    
    Returns:
    trend_series: ndarray, shape (n_samples, n_features)
    seasonal_series: ndarray, shape (n_samples, n_features)
    """
    trend_series = np.zeros(series.shape)
    seasonal_series = np.zeros(series.shape)

    for i in range(series.shape[1]):
        # 对每一列进行STL分解
        stl = STL(series[:, i],period=50,robust=False)
        res = stl.fit()

        # 获取趋势项和季节项
        trend_series[:, i] = res.trend
        seasonal_series[:, i] = res.seasonal

    return trend_series, seasonal_series

def loadData(args):
    # Traffic
    # Traffic = np.squeeze(np.load(args.traffic_file)['result'], -1)
    Traffic = np.load(args.traffic_file)['data']
    # 将 -1e9 值替换为 0
    # Traffic[Traffic == -1e9] = 0
    # bonus_all = np.load('/root/test/stock-STWave/data/STOCK/bonus.npy')
    path = '/root/autodl-tmp/Stockformer/Stockformer_run/us_alpha158_21-23_processed'
    files = os.listdir(path)
    data_list = []
    for file in files:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path, index_col=0)
        # 扩展维度
        arr = np.expand_dims(df.values, axis=2)
        data_list.append(arr)

    # 在第三个维度上拼接
    concatenated_arr = np.concatenate(data_list, axis=2)
    bonus_all = concatenated_arr

    print('target value shape:', Traffic.shape)
    print('extra value shape:', bonus_all.shape)
    infea = bonus_all.shape[-1]+1
    # train/val/test 
    num_step = Traffic.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = Traffic[: train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]

    # 趋势项和季节项提取
    ## train
    trend_train, seasonal_train = stl_decomposition(train)
    ## val
    trend_val, seasonal_val = stl_decomposition(val)
    ## test
    trend_test, seasonal_test = stl_decomposition(test)

    # bonus_all train/val/test
    bonus_all_train = bonus_all[: train_steps]
    bonus_all_val = bonus_all[train_steps : train_steps + val_steps]
    bonus_all_test = bonus_all[-test_steps :]
    # X, Y
    trainX, trainY = seq2instance(train, args.T1, args.T2)
    valX, valY = seq2instance(val, args.T1, args.T2)
    testX, testY = seq2instance(test, args.T1, args.T2)
    ## train
    trainXL, trainYL = seq2instance(trend_train, args.T1, args.T2)
    trainXH, trainYH = seq2instance(seasonal_train, args.T1, args.T2)
    ## val
    valXL, valYL = seq2instance(trend_val, args.T1, args.T2)
    valXH, valYH = seq2instance(seasonal_val, args.T1, args.T2)    
    ## test
    testXL, testYL = seq2instance(trend_test, args.T1, args.T2)
    testXH, testYH = seq2instance(seasonal_test, args.T1, args.T2)      


    # bonus_all X,Y
    bonus_all_trainX, bonus_all_trainY = bonus_seq2instance(bonus_all_train, args.T1, args.T2)
    bonus_all_valX, bonus_all_valY = bonus_seq2instance(bonus_all_val, args.T1, args.T2)
    bonus_all_testX, bonus_all_testY = bonus_seq2instance(bonus_all_test, args.T1, args.T2)
    
    
    
    # # disentangling
    # trainXL, trainXH = disentangle(trainX, args.w, args.j)
    # trainYL, trainYH = disentangle(trainY, args.w, args.j)
    # valXL, valXH = disentangle(valX, args.w, args.j)
    # testXL, testXH = disentangle(testX, args.w, args.j)
    # normalization
    # mean, std = np.mean(trainX), np.std(trainX)
    # trainXL, trainXH = (trainXL - mean) / std, (trainXH - mean) / std
    # valXL, valXH = (valXL - mean) / std, (valXH - mean) / std
    # testXL, testXH = (testXL - mean) / std, (testXH - mean) / std
    # trainX, valX, testX = (trainX - mean) / std, (valX - mean) / std, (testX - mean) / std
    # bonus_all normalization
    bonus_all_mean, bonus_all_std = np.mean(bonus_all_trainX), np.std(bonus_all_trainX)
    bonus_all_trainX = (bonus_all_trainX - bonus_all_mean) / bonus_all_std
    bonus_all_valX = (bonus_all_valX - bonus_all_mean) / bonus_all_std
    bonus_all_testX = (bonus_all_testX - bonus_all_mean) / bonus_all_std
    # temporal embedding
    tmp = {'PeMSD3':6,'PeMSD4':1,'PeMSD7':1,'PeMSD8':5, 'PeMSD7L':2, 'PeMSD7M':2, 'MYDATA':1, 'STOCK':3}
    days = {'PeMSD3':7,'PeMSD4':7,'PeMSD7':7,'PeMSD8':7, 'PeMSD7L':5, 'PeMSD7M':5, 'MYDATA':4, 'STOCK':5}
    TE = np.zeros([num_step, 2])
    startd = (tmp[args.Dataset] - 1) * 50
    df = days[args.Dataset]
    startt = 0
    for i in range(num_step):
        TE[i,0] = startd //  50
        startd = (startd + 1) % (df * 50)
        TE[i,1] = startt
        startt = (startt + 1) % 50
    ##############


    # train/val/test
    train = TE[: train_steps]
    val = TE[train_steps : train_steps + val_steps]
    test = TE[-test_steps :]
    # shape = (num_sample, P + Q, 2)
    trainTE = seq2instance(train, args.T1, args.T2)
    trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
    valTE = seq2instance(val, args.T1, args.T2)
    valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
    testTE = seq2instance(test, args.T1, args.T2)
    testTE = np.concatenate(testTE, axis = 1).astype(np.int32)
    
    return trainXL, trainXH, trainTE, trainY, trainYL, valXL, valXH, valTE, valY, testXL, testXH, testTE, testY, bonus_all_trainX, bonus_all_valX, bonus_all_testX, infea