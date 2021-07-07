import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Custom Ecg Dataset
class CustomEcgDataset(Dataset):
    def __init__(self):
        label_np = np.load('/Data_preprocessing/Data.npy')  # dtype:float64  shape:(100687, 251) <class 'numpy.ndarray'>
        data_np = np.load('/Data_preprocessing/Label.npy')  # dtype:float64  shape:(100687, ) <class 'numpy.ndarray'>
        self.Data = torch.from_numpy(label_np).float()  #RuntimeError: expected scalar type Float but found Double
        self.Label = torch.from_numpy(data_np).long()   #RuntimeError: expected scalar type Long but found Int

    def __getitem__(self, idx):
        label = self.Label[idx]
        data = self.Data[idx]
        sample = [data, label]
        return sample

    def __len__(self):
        return len(self.Label)

labels_map = {
    0: "Non ectopic",   #,非异位
    1: "Supraventriculr ectopic beat",  #,室上直肠搏动
    2: "Ventricular ectopic beat",  #,心室异位搏动
    3: "Fusion beat",   #,融合拍
    4: "Unknown beat",  #,未知节拍
}
full_dataset = CustomEcgDataset()
# print(len(full_dataset))
# data, label = full_dataset[100671]
# print(labels_map[label])
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
# print(len(full_dataset))
for i in range(1, cols*rows+1):
    sample_idx = torch.randint(len(full_dataset)-1, size=(1,)).item()
    print(sample_idx)
    data, label = full_dataset[sample_idx]
    # print(data.dtype)
    # print(label.dtype)
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label.item()])       # KeyError: tensor(0, dtype=torch.int32)
    # plt.axis("off")
    plt.plot(data)
plt.show()

#54813,71067我看都觉得不正常

batch_size = 64

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# for x, y in test_dataloader:
#     print(f"x.shape:{x.shape}")                         #x.shape:torch.Size([64, 251])
#     print(f"y.shape, y.dtype:{y.shape ,y.dtype}")       #y.shape, y.dtype:(torch.Size([64]), torch.int32)

#Creating Models
device = "cuda" if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(251, 12*12),
            nn.ReLU(),
            nn.Linear(12 * 12, 12 * 12),
            nn.ReLU(),
            nn.Linear(12*12, 4),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

#Optimizing the Model Parameters
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # print(X.dtype)
        # print(y.dtype)
        # X = X.float()
        # y = y.long()
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

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
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")

# #Saving Models
# torch.save(model.state_dict
# (), 'model.pth')
# print("Saved Pytorch Model State to model.pth")
#
# #Loading Models
# model = NeuralNetwork()
# model.load_state_dict(torch.load("model.pth"))
#
# #Test Models
# labels_map = {
#     0: "Non ectopic",   #,非异位
#     1: "Supraventriculr ectopic beat",  #,室上直肠搏动
#     2: "Ventricular ectopic beat",  #,心室异位搏动
#     3: "Fusion beat",   #,融合拍
#     4: "Unknown beat",  #,未知节拍
# }
#
# model.eval()
# x, y = test_data[0][0], test_data[0][1]
# with torch.no_grad():
#     pred = model(x)
#     predicted, actual = classes[pred[0].argmax(0)], classes[y]
#     print(f'Predicted: "{predicted}", Actual: "{actual}"')
