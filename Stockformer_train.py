from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import configparser
import math
import csv
import random
from pytorch_wavelets import DWT1DForward, DWT1DInverse


from lib import smallStockutils
from lib.smallStockutils import log_string, loadData, _compute_loss, metric
from lib.graph_utils import loadGraph
from Stockformer_run.Stockformer_code.Stockformermodel.Stockformermodels import Stockformer

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config)

parser.add_argument('--cuda', type=str, 
            default=config['train']['cuda'])
parser.add_argument('--seed', type = int, 
            default = config['train']['seed'])
parser.add_argument('--batch_size', type = int, 
            default = config['train']['batch_size'])
parser.add_argument('--max_epoch', type = int, 
            default = config['train']['max_epoch'])
parser.add_argument('--learning_rate', type=float, 
            default = config['train']['learning_rate'])

parser.add_argument('--Dataset', default = config['data']['dataset'])
parser.add_argument('--T1', type = int, 
            default = config['data']['T1'])
parser.add_argument('--T2', type = int, 
            default = config['data']['T2'])
parser.add_argument('--train_ratio', type = float, 
            default = config['data']['train_ratio'])
parser.add_argument('--val_ratio', type = float, 
            default = config['data']['val_ratio'])
parser.add_argument('--test_ratio', type = float, 
            default = config['data']['test_ratio'])

parser.add_argument('--L', type = int,
            default = config['param']['layers'])
parser.add_argument('--h', type = int,
            default = config['param']['heads'])
parser.add_argument('--d', type = int, 
            default = config['param']['dims'])
parser.add_argument('--j', type = int, 
            default = config['param']['level'])
parser.add_argument('--s', type = float,
            default = config['param']['samples'])
parser.add_argument('--w',
            default = config['param']['wave'])

parser.add_argument('--traffic_file', default = config['file']['traffic'])
parser.add_argument('--adj_file', default = config['file']['adj'])
parser.add_argument('--adjgat_file', default = config['file']['adjgat'])
parser.add_argument('--model_file', default = config['file']['model'])
parser.add_argument('--log_file', default = config['file']['log'])

args = parser.parse_args()

log = open(args.log_file, 'w')

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

def res(model, valXL, valXH, valTE, valY, bonus_all_valX, adjgat):
    model.eval()
    num_val = valXL.shape[0]
    num_batch = math.ceil(num_val / args.batch_size)

    pred = []
    label = []

    with torch.no_grad():
        for batch_idx in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

                xl = torch.from_numpy(valXL[start_idx : end_idx]).float().to(device)
                xh = torch.from_numpy(valXH[start_idx : end_idx]).float().to(device)
                bonus = torch.from_numpy(bonus_all_valX[start_idx : end_idx]).float().to(device)
                y = valY[start_idx : end_idx]
                te = torch.from_numpy(valTE[start_idx : end_idx]).to(device)

                y_hat, y_hat_l = model(xl, xh, te, bonus, adjgat)

                pred.append(y_hat.cpu().numpy())
                label.append(y)
    
    pred = np.concatenate(pred, axis = 0)
    label = np.concatenate(label, axis = 0)

    maes = []
    rmses = []
    mapes = []
    accuracys = []

    for i in range(pred.shape[1]):
        mae, rmse , mape, accuracy = metric(pred[:,i,:], label[:,i,:])
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        accuracys.append(accuracy)
        log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f, accuracy: %.4f' % (i+1, mae, rmse, mape, accuracy))
    
    mae, rmse, mape, accuracy = metric(pred, label)
    maes.append(mae)
    rmses.append(rmse)
    mapes.append(mape)
    accuracys.append(accuracy)
    log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f, accuracy: %.4f' % (mae, rmse, mape, accuracy))
    
    return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0), np.stack(accuracys, 0)

def test_res(model, valXL, valXH, valTE, valY, bonus_all_valX, adjgat):
    model.eval()
    num_val = valXL.shape[0]
    num_batch = math.ceil(num_val / args.batch_size)

    pred = []
    label = []

    with torch.no_grad():
        for batch_idx in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

                xl = torch.from_numpy(valXL[start_idx : end_idx]).float().to(device)
                xh = torch.from_numpy(valXH[start_idx : end_idx]).float().to(device)
                bonus = torch.from_numpy(bonus_all_valX[start_idx : end_idx]).float().to(device)
                y = valY[start_idx : end_idx]
                te = torch.from_numpy(valTE[start_idx : end_idx]).to(device)

                y_hat, y_hat_l = model(xl, xh, te, bonus, adjgat)

                pred.append(y_hat.cpu().numpy())
                label.append(y)
    
    pred = np.concatenate(pred, axis = 0)
    label = np.concatenate(label, axis = 0)

    maes = []
    rmses = []
    mapes = []
    accuracys = []

    for i in range(pred.shape[1]):
        mae, rmse , mape, accuracy = metric(pred[:,i,:], label[:,i,:])
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        accuracys.append(accuracy)
        log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f, accuracy: %.4f' % (i+1, mae, rmse, mape, accuracy))
    
    mae, rmse, mape, accuracy = metric(pred, label)
    maes.append(mae)
    rmses.append(rmse)
    mapes.append(mape)
    accuracys.append(accuracy)
    log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f, accuracy: %.4f' % (mae, rmse, mape, accuracy))

    # 从 NumPy 数组中提取数据
    data = pred[:, 0, :]

    # 写入 CSV 文件
    file_path = '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/output/first_pred.csv'
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

    print(f"已保存数据到 {file_path}")

    # 从 NumPy 数组中提取数据
    data = pred[:, 1, :]

    # 写入 CSV 文件
    file_path = '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/output/second_pred.csv'
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

    print(f"已保存数据到 {file_path}")

    # 从 NumPy 数组中提取数据
    data = label[:, 0, :]

    # 写入 CSV 文件
    file_path = '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/output/first_label.csv'
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

    print(f"已保存数据到 {file_path}")

    # 从 NumPy 数组中提取数据
    data = label[:, 1, :]

    # 写入 CSV 文件
    file_path = '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/output/second_label.csv'
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

    print(f"已保存数据到 {file_path}")

    return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0), np.stack(accuracys, 0)

def train(model, trainXL, trainXH, trainTE, trainY, trainYL, valXL, valXH, valTE, valY, bonus_all_trainX, bonus_all_valX, adjgat):
    num_train = trainXL.shape[0]
    min_loss = 200.0
    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20,    
                                    verbose=False, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=2e-6, eps=1e-08)
    
    for epoch in tqdm(range(1,args.max_epoch+1)):
        model.train()
        train_l_sum, batch_count, start = 0.0, 0, time.time()
        permutation = np.random.permutation(num_train)
        trainXL = trainXL[permutation]
        trainXH = trainXH[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]
        trainYL = trainYL[permutation]
        bonus_all_trainX = bonus_all_trainX[permutation]
        num_batch = math.ceil(num_train / args.batch_size)

        with tqdm(total=num_batch) as pbar:
            for batch_idx in range(num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_train, (batch_idx + 1) * args.batch_size)

                xl = torch.from_numpy(trainXL[start_idx : end_idx]).float().to(device)
                xh = torch.from_numpy(trainXH[start_idx : end_idx]).float().to(device)
                y = torch.from_numpy(trainY[start_idx : end_idx]).float().to(device)
                yl = torch.from_numpy(trainYL[start_idx : end_idx]).float().to(device)
                te = torch.from_numpy(trainTE[start_idx : end_idx]).to(device)
                bonus = torch.from_numpy(bonus_all_trainX[start_idx : end_idx]).float().to(device)
                
                
                optimizer.zero_grad()

                y_hat, y_hat_l = model(xl, xh, te, bonus,adjgat)

                loss = _compute_loss(y, y_hat) + _compute_loss(yl, y_hat_l)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                
                train_l_sum += loss.cpu().item()
                batch_count += 1
                pbar.update(1)

        log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
              % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))

        mae, rmse, mape, accuracy = res(model, valXL, valXH, valTE, valY, bonus_all_valX, adjgat)
        lr_scheduler.step(mae[-1])
        if mae[-1] < min_loss:
            min_loss = mae[-1]
            torch.save(model.state_dict(), args.model_file)

def test(model, valXL, valXH, valTE, valY, bonus_all_testX, adjgat):
    try:
        model.load_state_dict(torch.load(args.model_file))
    except EOFError:
        print(f"Error: Unable to load model state dictionary from file {args.model_file}. File may be empty or corrupted.")
        return

    mae, rmse, mape, accuracy = test_res(model, valXL, valXH, valTE, valY, bonus_all_testX, adjgat)
    return mae, rmse, mape, accuracy


if __name__ == '__main__':
    log_string(log, "loading data....")
    trainXL, trainXH, trainTE, trainY, trainYL, valXL, valXH, valTE, valY, testXL, testXH, testTE, testY, bonus_all_trainX, bonus_all_valX, bonus_all_testX, infeature = loadData(args)
    # adj, graphwave = loadGraph(args)
    adjgat = loadGraph(args)
    adjgat = torch.from_numpy(adjgat).float().to(device)
    log_string(log, "loading end....")

    log_string(log, "constructing model begin....")
    
    model = Stockformer(infeature, args.h*args.d, args.L, args.h, args.d, args.s, args.T1, args.T2, device).to(device)
    log_string(log, "constructing model end....")

    log_string(log, "training begin....")
    train(model, trainXL, trainXH, trainTE, trainY, trainYL, valXL, valXH, valTE, valY, bonus_all_trainX, bonus_all_valX, adjgat)
    log_string(log, "training end....")

    log_string(log, "testing begin....")
    test(model, testXL, testXH, testTE, testY, bonus_all_testX, adjgat)
    log_string(log, "testing end....")
