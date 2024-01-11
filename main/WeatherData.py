from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
import torch 
# 自定义数据集
class WeatherDataset(Dataset):
    def __init__(self, df):
        super(WeatherDataset, self).__init__()
        self.df = df
    
        # 定义数据扩增方法
        self.transform = T.Compose([
            T.Resize(size=(340,340)),
            T.RandomCrop(size=(256, 256)),
            T.RandomRotation(10),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 修改均值和标准差
        ])

    def __getitem__(self, index):
        file_name = self.df['filename'].iloc[index]
        # print(file_name)
        img = Image.open(file_name)
        img = self.transform(img)
        return img,\
                torch.tensor(self.df['period'].iloc[index]),\
                torch.tensor(self.df['weather'].iloc[index])

    def __len__(self):
        return len(self.df)