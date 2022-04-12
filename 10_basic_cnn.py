from turtle import forward
from numpy import size
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import os

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])

train_dataset = datasets.MNIST(root='./dataset/mnist/',
                                train=True,
                                download=True,
                                transform=transform)
train_loader = DataLoader(train_dataset,
                                shuffle=True,
                                batch_size=batch_size,
                                num_workers=0)

test_dataset = datasets.MNIST(root='./dataset/mnist/',
                                train=False,
                                download=True,
                                transform=transform)
test_loader = DataLoader(test_dataset,
                                shuffle=False,
                                batch_size=batch_size)

print("train_dataset length: ", len(train_dataset))


# 采用CNN构建网络，替换全连接层
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # input x: (n, 1, 28, 28)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))     # -> (n, 10, 24, 24) -> (n, 10, 12, 12)
        x = F.relu(self.pooling(self.conv2(x)))     # -> (n, 20, 8, 8) -> (n, 20, 4, 4)
        x = x.view(batch_size, -1) # flatten        # -> (n, 320)
        x = self.fc(x)                              # -> (n, 10)
        return x

# Exercise 10-1
class Net2(torch.nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(20, 30, kernel_size=3, padding=1)
        self.pooling = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(270, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))     # -> (n, 10, 28, 28) -> (n, 10, 14, 14)
        x = F.relu(self.pooling(self.conv2(x)))     # -> (n, 20, 14, 14) -> (n, 20, 7, 7)
        x = F.relu(self.pooling(self.conv3(x)))     # -> (n, 30, 7, 7) -> (n, 30, 3, 3)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     # 处理设备
model.to(device)            # 将模型model移到device中
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# x = torch.randn((2, 1, 28, 28))
# y = model(x)
# print(x)
# print(type(x), x.size())
# print(type(y), y.size())
# os._exit(0)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)   # 输入输出也放到device
        optimizer.zero_grad()           # 梯度清0

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()     # 防止生成计算图
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)   # 输入输出也放到device
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %f %%' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()