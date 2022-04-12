import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

'''
    为何DataLoader中的num_workers>=1时，为何整个代码会重复执行？
    num_workers=1,2 比num_workers=0来得慢
'''
# xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
# x_data = torch.from_numpy(xy[:, :-1])
# y_data = torch.from_numpy(xy[:, [-1]])  # 注意[-1]
# print(y_data)
# print(y_data.size())

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
for param in model.parameters():
    print(type(param), param.size())
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


if __name__ == "__main__":
    dataset = DiabetesDataset('diabetes.csv.gz')
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=1)

    for epoch in range(100):
        for i, (x_data, y_data) in enumerate(train_loader, 0):
            # Forward
            y_pred = model(x_data)
            loss = criterion(y_pred, y_data)
            print(epoch, loss.item())

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Update
            optimizer.step()

