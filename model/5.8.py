import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np


# torch.manual_seed(1)    # reproducible

# -----Hyper Parameters------
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate

device = "cuda" if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))

labels_map = {
    0: "Non ectopic",   #,非异位
    1: "Supraventriculr ectopic beat",  #,室上直肠搏动
    2: "Ventricular ectopic beat",  #,心室异位搏动
    3: "Fusion beat",   #,融合拍
    4: "Unknown beat",  #,未知节拍
}

# -----Custom Ecg Dataset-------
class CustomEcgDataset(Dataset):
    def __init__(self):
        label_np = np.load('/Data_preprocessing/Data.npy')  # dtype:float64  shape:(100687, 251) <class 'numpy.ndarray'>
        data_np = np.load('/Data_preprocessing/Data.npy')  # dtype:float64  shape:(100687, ) <class 'numpy.ndarray'>
        self.Data = torch.from_numpy(label_np).float()  #RuntimeError: expected scalar type Float but found Double
        self.Label = torch.from_numpy(data_np).long()   #RuntimeError: expected scalar type Long but found Int

    def __getitem__(self, idx):
        label = self.Label[idx]
        data = self.Data[idx]
        sample = [data, label]
        return sample

    def __len__(self):
        return len(self.Label)

# ----随机9个示例-----
full_dataset = CustomEcgDataset()
print(len(full_dataset))
data, label = full_dataset[100671]
print(labels_map[label])

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
# print(len(full_dataset))
for i in range(1, cols*rows+1):
    sample_idx = torch.randint(len(full_dataset)-1, size=(1,)).item()
    print(sample_idx)
    data, label = full_dataset[sample_idx]
    # print(data.dtype)
    print(label.dtype)
    print(label.shape)
    print(label)
    print()
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label.item()])       # KeyError: tensor(0, dtype=torch.int32)
    # plt.axis("off")
    plt.plot(data)
plt.show()

# ----划分训练集和测试集-----
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 251)
            nn.Conv1d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=6,              # filter size
                #stride=1,                   # filter movement/step
            ),                              # output shape (16, 246)  ((251-6)/1+1)
            nn.ReLU(),                      # activation
            nn.MaxPool1d(kernel_size=2),    # choose max value in 1x2 area, output shape (16, 123)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 123)
            nn.Conv1d(16, 32, 6),     # output shape (32, 118)
            nn.ReLU(),                      # activation
            nn.MaxPool1d(2),                # output shape (32, 59)
            # stride=1,
        )
        self.out = nn.Linear(32 * 59, 5)   # fully connected layer, output 5 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),  -1)           # flatten the output of conv2 to (batch_size, 32 * 59)
        #m = nn.Softmax(dim=1)
        output = self.out(x)    #m(self.out(x))
        return output, x    # return x for visualization


cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

#Optimizing the Model Parameters
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # print(X.dtype)
        # print(y.dtype)
        # X = X.float()
        # y = y.long()
        X, y = X.to(device), y.to(device)

        X = torch.unsqueeze(X, dim=1).type(torch.FloatTensor)   ## shape from (50, 251) to (50, 1, 251)
        # Compute prediction error
        pred = model(X)
        loss = loss_func(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            # X = X.float()
            # y = y.long()
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_func(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(EPOCH):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, cnn, loss_func, optimizer)
    test(test_dataloader, cnn)
print("Done!")