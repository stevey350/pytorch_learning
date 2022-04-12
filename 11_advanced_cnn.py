from turtle import forward
from numpy import size
import torch
from torch import nn
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


class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_1 = nn.Conv2d(in_channels,16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)
        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1408, 512)
        self.fc2 = nn.Linear(512, 10)       # ssj add

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                 # ssj add 
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)           # ssj add

    def forward(self, x):
        in_size = x.size(0)                     # x: (n, 1, 28, 28)
        x = self.mp(F.relu(self.conv1(x)))      # --> (n, 16, 24, 24) --> (n, 16, 12, 12)
        x = self.rblock1(x)                     # --> (n, 16, 12, 12)
        x = self.mp(F.relu(self.conv2(x)))      # --> (n, 32, 8, 8) --> (n, 32, 4, 4)
        x = self.rblock2(x)                     # --> (n, 32, 4, 4)
        x = x.view(in_size, -1)                 # --> (n, 512)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                         # ssj add
        return x


model = Net2()
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
    for epoch in range(15):
        train(epoch)
        test()