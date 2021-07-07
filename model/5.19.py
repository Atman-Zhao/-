import os
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

labels_map = {
    0: "Normal ",                       #正常
    1: "Non ectopic",                   #非异位
    2: "Supraventriculr ectopic beat",  #室上直肠搏动
    3: "Ventricular ectopic beat",      #心室异位搏动
    4: "Fusion beat",                   #融合拍
}

# ----自定义 Ecg Dataset
class CustomEcgDataset_train(Dataset):
    def __init__(self):
        data_np = np.load('D:\\Users\zhao_\Desktop\graduation project\ECG_Classification\Data_preprocessing\\train_data.npy')  # dtype:float64  shape:(100687, ) <class 'numpy.ndarray'>
        label_np = np.load('D:\\Users\zhao_\Desktop\graduation project\ECG_Classification\Data_preprocessing\\train_label.npy')  # dtype:float64  shape:(100687, 251) <class 'numpy.ndarray'>
        # print(data_np.shape)
        # print(label_np.shape)
        # print(data_np.dtype)
        # print(label_np.dtype)
        self.Data = torch.from_numpy(data_np).float()  #RuntimeError: expected scalar type Float but found Double
        self.Label = torch.from_numpy(label_np).long()   #RuntimeError: expected scalar type Long but found Int

        #self.Data = self.Data.to(torch.long)
        print(self.Data.dtype)
        print(self.Label.dtype)
    def __getitem__(self, idx):
        data = self.Data[idx]
        label = self.Label[idx]
        sample = [data, label]
        return sample
    def __len__(self):
        return len(self.Label)

class CustomEcgDataset_test(Dataset):
    def __init__(self):
        data_np = np.load('D:\\Users\zhao_\Desktop\graduation project\ECG_Classification\Data_preprocessing\\test_data.npy')  # dtype:float64  shape:(100687, ) <class 'numpy.ndarray'>
        label_np = np.load('D:\\Users\zhao_\Desktop\graduation project\ECG_Classification\Data_preprocessing\\test_label.npy')  # dtype:float64  shape:(100687, 251) <class 'numpy.ndarray'>
        self.Data = torch.from_numpy(data_np).float()  #RuntimeError: expected scalar type Float but found Double
        self.Label = torch.from_numpy(label_np).long()  #RuntimeError: expected scalar type Long but found Int
    def __getitem__(self, idx):
        label = self.Label[idx]
        data = self.Data[idx]
        sample = [data, label]
        return sample
    def __len__(self):
        return len(self.Label)

class Net(nn.Module):   #(50, 1, 288)
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=4, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=6, stride=1, padding=3)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1152, 81)
        self.linear2 = nn.Linear(81, 5)

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)  #(50, 8, 144)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)  #(50, 16, 72)
        x = torch.flatten(x, start_dim = 1) #(50, 1152)
        x = self.linear2(self.linear1(x))
        # x = F.linear(F.linear(x, (torch.tensor(20), torch.tensor(81))), torch.tensor(5, 20))
        return x

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch ,(x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        x = torch.unsqueeze(x,1)
        pred = model(x)
        loss = loss_fn(pred, y)
        # print("pred:",pred.dtype)
        # print("loss:",loss.dtype)
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

train_dataset = CustomEcgDataset_train()
test_dataset = CustomEcgDataset_test()
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

EPOCH = 5
BATCH_SIZE = 50
LR = 0.01

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = "cuda" if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))

net=Net().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

for t in range(EPOCH):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, net, loss_func, optimizer)
    test(test_dataloader, net, loss_func)
print("Done!")

#Saving Models
# torch.save(net.state_dict(), '/content/drive/MyDrive/Colab Notebooks/model.pth')
# print("Saved Pytorch Model State to model.pth")