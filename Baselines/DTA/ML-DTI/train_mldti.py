# %%
import os
import pathlib

# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import math
import time
import numpy as np
import pickle
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import pandas as pd

from dataset import *
from network.ML_DTI import DTImodel
from utils import *
from config.config_dict import Config
from log.train_logger import TrainLogger

# %%
def get_cindex(Y, P):
    summ = 0.
    pair = 0
    
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if(Y[i] > Y[j]):
                    pair += 1
                    summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])
        
    if pair != 0:
        return summ/pair, pair
    else:
        return 0, pair
def mse(y, f):
    y = y.detach().cpu().numpy().flatten()
    f = f.detach().cpu().numpy().flatten()
    return np.mean((y - f)**2)

def rmse(y, f):
    y = y.detach().cpu().numpy().flatten()
    f = f.detach().cpu().numpy().flatten()
    return np.sqrt(np.mean((y - f)**2))

def mae(y, f):
    y = y.detach().cpu().numpy().flatten()
    f = f.detach().cpu().numpy().flatten()
    return np.mean(np.abs(y - f))

def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))
def mse_print(y,f):
    mse = ((y - f)**2)
    return mse


def r_squared_error(y_obs, y_pred):
    y_obs = y_obs.detach().cpu().numpy().flatten()
    y_pred = y_pred.detach().cpu().numpy().flatten()
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)

    mult = np.sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult ** 2

    y_obs_sq = np.sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = np.sum((y_pred - y_pred_mean) ** 2)

    return mult / (y_obs_sq * y_pred_sq)


def squared_error_zero(y_obs, y_pred):
    y_obs = y_obs.detach().cpu().numpy().flatten()
    y_pred = y_pred.detach().cpu().numpy().flatten()

    k = get_k(y_obs, y_pred)
    y_obs_mean = np.mean(y_obs)

    upp = np.sum((y_obs - (k * y_pred)) ** 2)
    down = np.sum((y_obs - y_obs_mean) ** 2)

    return 1 - (upp / down)


def rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.abs((r2 * r2) - (r02 * r02))))

def train(model, device, dataloader):
    model.train()
    for data in train_loader:
        # global_step += 1
        drug, target, label = data
        drug, target, label = drug.to(device), target.to(device), label.to(device)

        pred = model(target, drug)

        loss = criterion(pred.view(-1), label)
        cindex, pair = get_cindex(label.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return
def val(model, criterion, dataloader):
    model.eval()
    running_loss = AverageMeter()
    running_cindex = AverageMeter()

    for data in dataloader:
        drug, target, label = data
        drug, target, label = drug.to(device), target.to(device), label.to(device)

        with torch.no_grad():
            pred = model(target, drug)
            loss = criterion(pred.view(-1), label)

            # 计算各项指标
            mse_val = mse(label, pred)
            rmse_val = rmse(label, pred)
            mae_val = mae(label, pred)
            rm2_val = rm2(label, pred)

            cindex, pair = get_cindex(label.detach().cpu().numpy().reshape(-1),
                                      pred.detach().cpu().numpy().reshape(-1))

            running_loss.update(loss.item(), label.size(0))
            running_cindex.update(cindex, pair)

    epoch_loss = running_loss.get_average()
    epoch_cindex = running_cindex.get_average()
    running_loss.reset()
    running_cindex.reset()

    model.train()

    return epoch_loss, mse_val, rmse_val, epoch_cindex, rm2_val

# %%
# for fold in range(5):
# config = Config()
# args = config.get_config()
# args['fold'] = fold
# logger = TrainLogger(args)
# logger.info(__file__)
#
# data_root = args.get("data_root")
# DATASET = args.get("dataset")
# split_type = args.get("split_type")
# save_model = args.get("save_model")
# fold = args.get("fold")
#
# fpath = os.path.join(data_root, DATASET)
# dp = DataPrepared(fpath)
# train_index, val_index, test_index = dp.read_sets(fold, split_type=split_type)
# df = dp.get_data()
# train_df = df.iloc[train_index]
# val_df = df.iloc[val_index]
# test_df = df.iloc[test_index]
dataset = 'davis'
df_train = pd.read_csv(f'data/{dataset}/train.csv')# Reading training data.
df_test = pd.read_csv(f'data/{dataset}/test.csv') # Reading test data.

train_set = PairedDataset(df_train)
# val_set = PairedDataset(val_df)
test_set = PairedDataset(df_test)

train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=4)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)
device = torch.device(f'cuda:0' if torch.cuda.is_available() else "cpu")

model = DTImodel(26, 65).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
best_mse = 1000
epochs = 2000
steps_per_epoch = 200
# num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
break_flag = False

global_step = 0
global_epoch = 0
early_stop_epoch = 50

running_loss = AverageMeter()
running_cindex = AverageMeter()
running_best_mse = BestMeter("min")

model.train()
directory = f'results/{dataset}/'
pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
model_file_name = directory + f'train_{dataset}.model'
for epoch in range(epochs):
    train(model,device,train_loader)
    test_loss, test_mse,test_rmse,test_cindex,tese_r2 = val(model, criterion, test_loader)
    if test_mse < best_mse:
        torch.save(model.state_dict(), model_file_name)
        best_epoch = epoch + 1
        best_mse = test_mse
        best_rmse = test_rmse
        best_ci = test_cindex
        best_rm2 = tese_r2
        print('mse improved at epoch ', best_epoch, '; best_mse,best_rmse,best_ci,best_rm2:',
              best_mse, best_rmse, best_ci, best_rm2, dataset)
    else:
        print(test_mse, 'No improvement since epoch ', best_epoch, '; best_mse,best_rmse,best_ci,best_rm2:',
              best_mse, best_rmse, best_ci, best_rm2, dataset)


            

# %%


