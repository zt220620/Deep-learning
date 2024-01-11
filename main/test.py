import io
import math, json
import glob
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn
from WeatherData import WeatherDataset
from torch.utils.data import DataLoader
from rsnet18 import ResNet,BasicBlock
from my_plot import plot_test_loss,plot_test_accuracy
import matplotlib.pyplot  as plt
device = torch.device("cpu")

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
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

model=ResNet(BasicBlock,[2, 2, 2, 2]).to(device=device)

# 指定要加载的模型的训练轮数
load_epoch = 50

# 构建加载模型的路径
model_load_path = f'./models2/saving_{load_epoch}.pth'

# 创建模型和优化器（确保与保存时一致）
model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device=device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
#交叉熵损失
criterion = torch.nn.CrossEntropyLoss()
# 加载模型检查点
checkpoint = torch.load(model_load_path)
# 还原模型和优化器状态
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 测试集数据路径
test_df = pd.DataFrame({'filename': glob.glob('./test_images/*.jpg')})
test_df['period'] = 0
test_df['weather'] = 0
test_df = test_df.sort_values(by='filename')

test_dataset = WeatherDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model.eval()
correct_period = 0
total_period = 0
correct_weather = 0
total_weather = 0
period_pred = []
weather_pred = []
period,weather=[],[]
ACC1,ACC2=[],[]
sum_loss=[]
# 测试集进行预测
for i, (x, y1, y2) in enumerate(test_loader):
    pred1, pred2 = model(x)
    loss1=criterion(pred1, y1)
    loss2=criterion(pred2, y1)
    _, predicted_period = torch.max(pred1.data, 1)
    _, predicted_weather = torch.max(pred2.data, 1)

    total_period += y1.size(0)
    correct_period += (predicted_period == y1).sum().item()
    ACC1.append((predicted_period == y1).sum().item())
    total_weather += y2.size(0)
    correct_weather += (predicted_weather == y2).sum().item()
    ACC2.append((predicted_weather == y2).sum().item())
    period.append(loss1.item())
    weather.append(loss2.item())
    sum_loss.append(loss1.item()+loss2.item())
    period_pred += period_dict[pred1.argmax(1).numpy()].tolist()
    weather_pred += weather_dict[pred2.argmax(1).numpy()].tolist()

accuracy_period = correct_period / total_period
accuracy_weather = correct_weather / total_weather

print(f'准确率 - 时间: {accuracy_period * 100:.2f}%')
print(f'准确率 - 天气: {accuracy_weather * 100:.2f}%')
plot_test_loss(period,"时间")
plot_test_loss(weather,"天气")
plot_test_loss(sum_loss,"总")
plot_test_accuracy(ACC1,"时间")
plot_test_accuracy(ACC2,"天气")
test_df['period'] = period_pred
test_df['weather'] = weather_pred

submit_json = {
    'annotations':[]
}

# 生成测试集结果
for row in test_df.iterrows():
    submit_json['annotations'].append({
        'filename':  row[1].filename.split('/')[-1],
        'period': row[1].period,
        'weather': row[1].weather,
    })

with open('submit.json', 'w') as up:
    json.dump(submit_json, up)