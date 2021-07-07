import os
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

label s_map = {
    0: "Normal ",                       #正常
    1: "Non ectopic",                   #非异位
    2: "Supraventriculr ectopic beat",  #室上直肠搏动
    3: "Ventricular ectopic beat",      #心室异位搏动
    4: "Fusion beat",                   #融合拍
}

# ----自定义 Ecg Dataset
class CustomEcgDataset(Dataset):
    def __init__(self, dir, train):
        if train == True:
          path_data = dir + 'train_data.npy'
          path_label = dir + 'train_label.npy'
          self.data = torch.from_numpy(np.load(path_data)).float()
          self.label = torch.from_numpy(np.load(path_label)).long()
        if train == False:
          path_data = dir + 'test_data.npy'
          path_label = dir + 'test_label.npy'
          self.data = torch.from_numpy(np.load(path_data)).float()
          self.label = torch.from_numpy(np.load(path_label)).long()

    def __getitem__(self, idx):
        label = self.label[idx]
        data = self.data[idx]
        sample = [data, label]
        return sample

    def __len__(self):
        return len(self.label)

class Net(nn.Module):   #(50, 1, 324)
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=4, stride=1, padding=2) #L_out=324
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=6, stride=1, padding=3) #L_out=162
        self.fc1 = nn.Linear(16*81, 512)
        self.fc2 = nn.Linear(512, 81)
        self.fc3 = nn.Linear(81, 5)

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)  #(50, 8, 162)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)  #(50, 16, 81)
        x = torch.flatten(x, 1) # 展平除批处理的所有维度
        x = F.relu(self.fc1(x)) #(50, 512)
        x = F.relu(self.fc2(x)) #(50, 81)
        x = self.fc3(x) #(50, 5)
        return x

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch ,(x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        x = torch.unsqueeze(x,1)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch ,(x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            x = torch.unsqueeze(x, 1)
            pred = model(x)
            loss_fn(pred, y)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
        test_loss /= size
        correct /= size
        print(f" Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def test_each_class(dataloader, model):
  # 准备计算每个类的正确预测数
  correct_pred = {labels_map[classname]: 0 for classname in labels_map}
  total_pred = {labels_map[classname]: 0 for classname in labels_map}

  with torch.no_grad():
    for x, y in dataloader:
      x, y = x.to(device), y.to(device)
      x = torch.unsqueeze(x, 1)
      outputs = model(x)
      for label, prediction in zip(y, outputs.argmax(1)):
        if label == prediction:
          correct_pred[labels_map[label.item()]] += 1
        total_pred[labels_map[label.item()]] += 1
  #输出每个类的准确率
  for classname , correct_count in correct_pred.items():
    accuracy = 100* float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,accuracy))

EPOCH = 5
BATCH_SIZE = 100
LR = 0.01

# ----随机划分训练集和测试集
train_dataset = CustomEcgDataset('/content/drive/MyDrive/Colab Notebooks/ecg/data_preprocessing/', True)
test_dataset = CustomEcgDataset('/content/drive/MyDrive/Colab Notebooks/ecg/data_preprocessing/', False)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

device = "cuda" if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))

net=Net().to(device)
print(net)
optimizer = torch.optim.Adam(net.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# ----随机9个样本
figure = plt.figure(figsize=(10, 10))
cols, rows = 3, 3
# print(len(full_dataset))
for i in range(1, cols*rows+1):
    sample_idx = torch.randint(len(train_dataset)-1, size=(1,)).item()
    data, label = train_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label.item()])       # KeyError: tensor(0, dtype=torch.int32)
    #print(data.shape)
    plt.plot(data)
plt.show()

for t in range(EPOCH):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, net, loss_func, optimizer)
    test(test_dataloader, net, loss_func)
print("Done!")
test_each_class(test_dataloader, net)

#Saving Models
# torch.save(net.state_dict(), '/content/drive/MyDrive/Colab Notebooks/model.pth')
# print("Saved Pytorch Model State to model.pth")