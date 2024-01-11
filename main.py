import io
import math, json
import glob
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn
import matplotlib.pyplot as plt
from WeatherData import WeatherDataset
from torch.utils.data import DataLoader
from rsnet18 import ResNet,BasicBlock
from my_plot import plot_train_accuracy,plot_train_loss
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 读取数据集标注，提取标注信息中的关键信息
train_json = pd.read_json('train.json')
train_json['filename'] = train_json['annotations'].apply(lambda x: x['filename'].replace('\\', '/'))
train_json['period'] = train_json['annotations'].apply(lambda x: x['period'])
train_json['weather'] = train_json['annotations'].apply(lambda x: x['weather'])

train_json['period'], period_dict = pd.factorize(train_json['period'])
train_json['weather'], weather_dict = pd.factorize(train_json['weather'])


# 训练集
train_dataset = WeatherDataset(train_json.iloc[:-500])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 验证集
val_dataset = WeatherDataset(train_json.iloc[-500:])
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

model=ResNet(BasicBlock,[2, 2, 2, 2]).to(device=device)
# 定义损失函数和优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001)
#交叉熵损失
criterion = torch.nn.CrossEntropyLoss()
Train_Loss, Val_Loss = [], []
Train_ACC1, Train_ACC2 = [], []
Val_ACC1, Val_ACC2 = [], []
loss1,loss2=[],[]
for epoch in range(50):
    # 模型训练
    model.train()
    for i, (x, y1, y2) in enumerate(train_loader):
        x=x.to(device)
        y1=y1.to(device)
        y2=y2.to(device)
        pred1, pred2 = model(x)
        # 类别1 loss + 类别2 loss 为总共的loss
        period_loss=criterion(pred1, y1)
        weather_loss=criterion(pred2, y2)
        loss = criterion(pred1, y1) + criterion(pred2, y2)
        Train_Loss.append(loss.item())
        loss1.append(period_loss.item())
        loss2.append(weather_loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 将预测值和标签移动到 CPU 上，然后再转换为 NumPy 数组
        Train_ACC1.append((pred1.argmax(1).cpu() == y1.flatten().cpu()).numpy().mean())
        Train_ACC2.append((pred2.argmax(1).cpu() == y2.flatten().cpu()).numpy().mean())

        Train_ACC1[-1] = 0.0 if math.isnan(Train_ACC1[-1]) else Train_ACC1[-1]
        Train_ACC2[-1] = 0.0 if math.isnan(Train_ACC2[-1]) else Train_ACC2[-1]

    # 模型验证
    model.eval()
    for i, (x, y1, y2) in enumerate(val_loader):
        x=x.to(device)
        y1=y1.to(device)
        y2=y2.to(device)
        pred1, pred2 = model(x)
        loss = criterion(pred1, y1) + criterion(pred2, y2)
        Val_Loss.append(loss.item())
         # 将预测值和标签移动到 CPU 上，然后再转换为 NumPy 数组
        Val_ACC1.append((pred1.argmax(1).cpu() == y1.flatten().cpu()).numpy().mean())
        Val_ACC2.append((pred2.argmax(1).cpu() == y2.flatten().cpu()).numpy().mean())

        Val_ACC1[-1] = 0.0 if math.isnan(Val_ACC1[-1]) else Val_ACC1[-1]
        Val_ACC2[-1] = 0.0 if math.isnan(Val_ACC2[-1]) else Val_ACC2[-1]

    if (epoch+1) % 5 == 0:
        plot_train_loss(Train_Loss,"总共")
        plot_train_loss(loss1,"时间")
        plot_train_loss(loss2,"天气")
        plot_train_accuracy(Val_ACC1,"时间")
        plot_train_accuracy(Val_ACC1,"天气")
        print(f'\nEpoch: {epoch+1}')
        print(f'Loss {np.mean(Train_Loss):3.5f}/{np.mean(Val_Loss):3.5f}')
        print(f'period.ACC {np.mean(Train_ACC1):3.5f}/{np.mean(Val_ACC1):3.5f}')
        print(f'weather.ACC {np.mean(Train_ACC2):3.5f}/{np.mean(Val_ACC2):3.5f}')

        # 保存模型检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': np.mean(Train_Loss),
            'val_loss': np.mean(Val_Loss),
            'train_period_acc': np.mean(Train_ACC1),
            'val_period_acc': np.mean(Val_ACC1),
            'train_weather_acc': np.mean(Train_ACC2),
            'val_weather_acc': np.mean(Val_ACC2)
        }
        torch.save(checkpoint, f"./models2/saving_{epoch+1}.pth")
        print("保存成功!")
