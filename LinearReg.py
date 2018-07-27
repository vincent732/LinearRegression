#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 18:22:33 2017

@author: vincent
"""
from pandas import read_csv
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def error(label, row, weights):
    a = np.matrix(weights)
    b = np.matrix(row.values)
    return label - np.asscalar(np.inner(a, b))


# y = w0 * x0 + w1 * x1 + w2 * w2 + ....+ w9 * x9
def compute_loss(weights, data, label):
    N = float(len(data))
    return sum([error(label[index], row, weights)**2 for index, row in data.iterrows()])/N


def step_gradient_descent(w, x, y, eta, londa):
    gradient = [0] * 163
    N = float(len(x))
    for j, row in x.iterrows():
        err = error(y[j], row, w)
        for i in range(len(w)):
            gradient[i] += -(2/N) * row[i - 1] * err

    # regularizer
    for i in range(len(gradient)):
        gradient[i] += 2 * londa * w[i]
    return [w[i] - (eta * gradient[i]) for i in range(len(gradient))]


def data_pre_processing():
    raw = read_csv("/home/vincent/Public/data/train.csv", header = 0)
     # replace all NR to 0.0 and change type to float
    transformation = lambda x : 0.0 if x == "NR" else float(x)
    date_group = raw.groupby(['date'])

    previous_month = -1
    month_data = pd.DataFrame()
    X = pd.DataFrame()
    Y = pd.DataFrame()
    forceEnd = False
    for key in sorted(date_group.groups.keys()):
        current_month = key.split("/")[1]

        if previous_month == -1:
            previous_month = current_month

        # +++ handle single day data
        single_day = date_group.get_group(key).iloc[:, 3:].applymap(transformation).reset_index(drop=True)
        old_columns = [str(x) for x in range(24)]
        new_columns = range(len(month_data.columns) , len(month_data.columns) + 24)
        single_day = single_day.rename(columns = dict(zip(old_columns, new_columns)))
        # ---
        if key == '2014/12/20':
            month_data = pd.concat([month_data, single_day], axis=1, ignore_index=True)
            forceEnd = True
        if previous_month != current_month or forceEnd: # if switch to next month or the end of this data
            i = 0
            while i <= 470:
                x = month_data.iloc[:, i : i + 9]
                y = month_data.iloc[9:10, i + 9: i + 10]
                x = pd.DataFrame(x.values.reshape(1, -1))
                y = pd.DataFrame(y.values.reshape(1, -1))
                X = X.append(x, ignore_index=True)
                Y = Y.append(y, ignore_index=True)
                i += 1
            month_data = pd.DataFrame()#  regenerate another dataframe for next month

        if forceEnd == False:
            month_data = pd.concat([month_data, single_day], axis=1, ignore_index=True)
            previous_month = current_month


    X = X.reset_index(drop=True)
    Y = Y.reset_index(drop=True)
    X.to_csv("/home/vincent/Public/X.csv", header=range(0,162),index=False)
    Y.to_csv("/home/vincent/Public/Y.csv", header=['label'],index=False)


def training(londas, learning_rate, data, label, iterations = 3000):
    # transfer to list
    label = label['label'].tolist()
    # initial parameters
    data.insert(0, -1, 1)
    all_minError = []
    all_weights = []
    for londa in londas:
        weights = [0] * 163
        min_error = float("inf")
        for i in range(iterations):
            print("===== Iterations " + str(i) + " with londa:" + str(londa))
            total_loss = compute_loss(weights, data, label)
            print("Total loss:" + str(total_loss))
            if min_error < total_loss:
                break
            else:
                min_error = total_loss
            weights = step_gradient_descent(weights, data, label, learning_rate, londa)
        all_minError.append(min_error)
        all_weights.append(weights)
    plt.xlabel("Learning rates")
    plt.ylabel("Loss")
    plt.plot(londas, all_minError)
    plt.savefig("/home/vincent/Public/eta_vs_loss.png")
    return all_weights


def testing(weights, index):
    print("=========================TESTING================================")
    column  = ['id', 'standard', '0', '1', '2', '3', '4', '5', '6', '7', '8']
    raw = read_csv("/home/vincent/machine_learning/LinearReg/data/test_X.csv", 
                   header = 0)
    raw.columns = column
    id_group = raw.groupby(['id'])
    transformation = lambda x : 0.0 if x == "NR" else float(x)
    data = pd.DataFrame()
    for key in id_group.groups.keys():
        row = id_group.get_group(key).iloc[:, 2:].applymap(transformation)
        row = pd.DataFrame(row.values.reshape(1, -1))
        #  insert a x0 vector
        bias = pd.DataFrame([1], columns=['-1'])
        # add to x vector
        row = pd.concat([bias, row], axis=1, ignore_index = True)
        # add to testing features
        data = pd.concat([data,row])
    data = data.reset_index(drop=True)
    print(data)
    #  generate prediction
    column = ["id", "value"]
    output = pd.DataFrame(columns = column)
    for i, row in data.iterrows():
        a = np.matrix(weights)
        b = np.matrix(row.values)
        value = ["id_" + str(i), np.asscalar(np.inner(a, b))]
        output = output.append(dict(zip(column, value)), ignore_index=True)
    output.to_csv("/home/vincent/Public/"+ str(index) +".csv", index=False)

#data_pre_processing()
data = read_csv("/home/vincent/Public/X.csv", index_col=None)
label = read_csv("/home/vincent/Public/Y.csv", index_col=None)
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
for train_index, validation_index in kf.split(data.values):
    print(data.values[train_index])
    print(data.values[validation_index])

'''
londa = [1, 10, 100, 1000, 10000]
weights = training(londa, 1.0/(10**6))
i = 0
for weight in weights:
    testing(weight, londa[i])
    i+=1
'''