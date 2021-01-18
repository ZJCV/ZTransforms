# -*- coding: utf-8 -*-

# @Time    : 19-4-16 上午10:04
# @Author  : zj

"""
最小二乘法计算线性回归问题
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_sweden_data():
    """
    加载单变量数据
    """
    path = 'data/sweden.txt'
    x = []
    y = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip().split('\t')
            x.append(int(item[0]))
            y.append(float(item[1]))

    return np.array(x), np.array(y)


def load_ex1_multi_data():
    """
    加载多变量数据
    """
    path = '../data/coursera2.txt'
    datas = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            datas.append(line.strip().split(','))
    data_arr = np.array(datas)
    data_arr = data_arr.astype(np.float)

    X = data_arr[:, :2]
    Y = data_arr[:, 2]
    return X, Y


def least_square_loss_v1(x, y):
    """
    最小二乘法，几何运算
    """
    X = np.array(x)
    Y = np.array(y)
    muX = np.mean(X)
    muY = np.mean(Y)
    muXY = np.mean(X * Y)
    muXX = np.mean(X * X)

    w1 = (muXY - muX * muY) / (muXX - muX ** 2)
    w0 = muY - w1 * muX
    return w0, w1


def least_square_loss_v2(x, y):
    """
    最小二乘法，矩阵运算
    """
    extend_x = np.insert(x, 0, values=np.ones(x.shape[0]), axis=1)
    w = np.linalg.inv(extend_x.T.dot(extend_x)).dot(extend_x.T).dot(y)
    return w


def least_square_loss_v3(x, y):
    """
    最小二乘法，矩阵运算，PyTorch实现
    """
    x = np.vstack([x, np.ones(len(x))]).T
    A = torch.from_numpy(x).float()
    print(A.shape)
    w = torch.lstsq(torch.from_numpy(y).float(), A)[0]
    return w


def compute_single_variable_linear_regression():
    x, y = load_sweden_data()
    # w0, w1 = least_square_loss_v1(x, y)
    w0, w1 = least_square_loss_v3(x, y)

    y2 = w1 * x + w0

    plt.scatter(x, y)
    plt.plot(x, y2)

    plt.show()


def compute_multi_variable_linear_regression():
    x, y = load_ex1_multi_data()
    # 计算权重
    w = least_square_loss_v2(x, y)
    print(w)


if __name__ == '__main__':
    compute_single_variable_linear_regression()
    # compute_multi_variable_linear_regression()
