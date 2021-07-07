#Loading Models
net_load = Net()
net_load.load_state_dict(torch.load("/content/drive/MyDrive/Colab Notebooks/model.pth"))

# ----随机9个样本
figure = plt.figure(figsize=(14, 14))
cols, rows = 3, 3
for i in range(1, cols*rows+1):
    sample_idx = torch.randint(len(full_dataset)-1, size=(1,)).item()
    data, label = full_dataset[sample_idx]
    data = torch.unsqueeze(data,0)
    data = torch.unsqueeze(data,0)

    with torch.no_grad():
      pred = net_load(data)
      print(pred.argmax(1))
    figure.add_subplot(rows, cols, i)
    plt.title(f"actual:{labels_map[label.item()]}")
    #plt.title(f"actual:{labels_map[label.item()]}\n pred:{labels_map[pred]}")
    #print(f"actual:{labels_map[label.item()]}\n pred:{labels_map[pred.to('cpu')]}")
    #print(data[0, 0, :].shape)
    plt.plot(data[0, 0, :])
plt.show()