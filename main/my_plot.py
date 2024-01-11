import torch
from matplotlib import pyplot as plt
import numpy as np
import torchvision
import matplotlib
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False    #显示负号
def plot_train_loss(data,titles):
    fig=plt.figure()
    plt.plot(range(len(data)),data,color='blue')
    plt.title("%s训练损失"%titles)
    plt.legend(['value'],loc='upper right')
    plt.xlabel('batch_x')
    plt.ylabel('loss')
    plt.show()

def plot_train_accuracy(data,titles):
    fig=plt.figure()
    plt.plot(range(len(data)),data,color='blue')
    plt.title("%s训练准确度"%titles)
    plt.legend(['value'],loc='upper right')
    plt.xlabel('batch_x')
    plt.ylabel('accuracy')
    plt.show()

def plot_train_mLoss(data):
    fig=plt.figure()
    plt.plot(range(len(data)),data,color='blue')
    plt.title("训练平均损失")
    plt.legend(['value'],loc='upper right')
    plt.xlabel('batch_x')
    plt.ylabel('mloss')
    plt.show()

def plot_test_accuracy(data,titles):
    fig=plt.figure()
    plt.plot(range(len(data)),data,color='blue')
    plt.title("%s测试准确度"%titles)
    plt.legend(['value'],loc='upper right')
    plt.xlabel('batch_x')
    plt.ylabel('accuracy')
    plt.show()

def plot_test_loss(data,titles):
    fig=plt.figure()
    plt.plot(range(len(data)),data,color='blue')
    plt.title("%s测试损失"%titles)
    plt.legend(['value'],loc='upper right')
    plt.xlabel('batch_x')
    plt.ylabel('loss')
    plt.show()