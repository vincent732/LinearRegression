#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 18:22:33 2017

@author: vincent
"""
from pandas import read_csv
import pandas as pd
import numpy as np


def error(y, x, w):
    return y - np.asscalar(np.inner(np.matrix(w), np.matrix(x.values)))


# y = w0 * x0 + w1 * x1 + w2 * w2 + ....+ w9 * x9
def compute_loss(w, x, l):
    return sum([error(row[163:].values[0], row[:163], w)**2 for index, row in x.iterrows()]) \
           + l * sum([weight**2 for weight in w])


def step_gradient_descent(w, x, eta, l):
    gradient = [0] * 163
    for row_index, row in x.iterrows():
        err = error(row[163:].values[0], row[:163], w)
        gradient = [gradient[weight_index] - 2 * row[weight_index] * err for weight_index in range(len(w))]

    gradient = [gradient[index] + 2 * l * w[index] for index in range(len(gradient))]

    return [w[i] - (eta * gradient[i]) for i in range(len(w))]


def data_pre_processing():
    raw = read_csv("/home/vincent/machine_learning/LinearReg/data/train.csv", header=0)
     # replace all NR to 0.0 and change type to float
    transformation = lambda x : 0.0 if x == "NR" else float(x)
    date_group = raw.groupby([raw.columns[0]])

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
        single_day = date_group.get_group(key).iloc[9:10, 3:].applymap(transformation).reset_index(drop=True)
        print(single_day)
        old_columns = [str(x) for x in range(24)]
        new_columns = range(len(month_data.columns) , len(month_data.columns) + 24)
        single_day = single_day.rename(columns = dict(zip(old_columns, new_columns)))
        # ---
        if key == '2014/12/20':
            month_data = pd.concat([month_data, single_day], axis=1, ignore_index=True)
            forceEnd = True
        if previous_month != current_month or forceEnd: # if switch to next month or the end of this data
            i = 0
            print(current_month)
            while i <= 470:
                x = month_data.iloc[:, i : i + 9]
                y = month_data.iloc[:, i + 9: i + 10]
                x = pd.DataFrame(x.values.reshape(1, -1))
                y = pd.DataFrame(y.values.reshape(1, -1))
                X = X.append(x, ignore_index=True)
                Y = Y.append(y, ignore_index=True)
                i += 1
            month_data = pd.DataFrame()# regenerate another dataframe for next month

        if forceEnd == False:
            month_data = pd.concat([month_data, single_day], axis=1, ignore_index=True)
            previous_month = current_month


    X = X.reset_index(drop=True)
    Y = Y.reset_index(drop=True)
    X.to_csv("/home/vincent/machine_learning/LinearReg/data/X.csv", header=range(0,9),index=False)
    Y.to_csv("/home/vincent/machine_learning/LinearReg/data/Y.csv", header=['label'],index=False)


def training(l, learning_rate, x, y):
    # initial parameters
    w = [0.2] * 163
    for iteration in range(10):
        print(str(iteration) + " epochs")
        training_data = pd.concat([x, y], axis=1, ignore_index=True)
        np.random.shuffle(training_data.values)
        for j in range(len(training_data)):
            total_loss = compute_loss(w, training_data, l)
            print("total_loss:" + str(total_loss))
            w = step_gradient_descent(w, training_data.iloc[j:j+1,], learning_rate, l)
    return w


def predict(w, x):
    output = pd.DataFrame(columns=["id", "value"])
    for i, row in x.iterrows():
        print(i)
        print(np.asscalar(np.inner(np.matrix(w), np.matrix(row.values))))
        value = ["id_" + str(i), np.asscalar(np.inner(np.matrix(w), np.matrix(row.values)))]
        output = output.append(dict(zip(["id", "value"], value)), ignore_index=True)
    return output

#data_pre_processing()
'''
data = read_csv("/home/vincent/Public/X.csv", index_col=None)
label = read_csv("/home/vincent/Public/Y.csv", index_col=None)
learning_rates_candidate = [1.0/10**7]
londa_candidate = [0]
result = [0] * 1

from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
for train_index, validation_index in kf.split(data.values):
    train_x = pd.DataFrame(data.values[train_index])
    train_x.insert(0, -1, 1)
    train_y = pd.DataFrame(label.values[train_index])
    validation_x = pd.DataFrame(data.values[validation_index])
    validation_x.insert(0, -1, 1)
    validation_y = pd.DataFrame(label.values[validation_index])
    validation = pd.concat([validation_x, validation_y], axis=1, ignore_index=True)
    for i in range(len(londa_candidate)):
        for j in range(len(learning_rates_candidate)):
            weights = training(londa_candidate[i], learning_rates_candidate[j], x=train_x, y=train_y)
            loss = compute_loss(weights, validation, londa_candidate[i])
            print("==== result with learning rate = " + str(learning_rates_candidate[j]) + " and londa:" + str(londa_candidate[i]) + "======")
            print("loss" + str(loss))
            result[(i+1)*(j+1) - 1] += loss
print(result)
data.insert(0, -1, 1)
weights = training(0, 1.0/10**7, data, label)
data = pd.concat([data, label], axis=1, ignore_index=True)
loss = compute_loss(weights, data, 0)
print(loss)
print(weights)
'''
######### transform raw data input features
column = ['id', 'standard', '0', '1', '2', '3', '4', '5', '6', '7', '8']
raw = read_csv("/home/vincent/machine_learning/LinearReg/data/test_X.csv", header=0)
raw.columns = column
id_group = raw.groupby(['id'])
transformation = lambda feature: 0.0 if feature == "NR" else float(feature)

X = pd.DataFrame()
for key in sorted(id_group.groups.keys()):
    print(id_group.get_group(key))
    row = id_group.get_group(key).iloc[9:10, 2:].applymap(transformation)
    # add to testing features
    X = pd.concat([X,row])
X.to_csv("/home/vincent/machine_learning/LinearReg/data/test_x.csv", index=False)
# = predict(weights, X)
#predicted_Y.to_csv("/home/vincent/machine_learning/LinearReg/data/output.csv", index=False)