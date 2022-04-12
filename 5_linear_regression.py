
import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])    # 3*1 matrix
y_data = torch.Tensor([[2.0], [4.0], [6.0]])    # 3*1 matrix

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1, bias=True)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()
criterion = torch.nn.MSELoss(size_average=False)
for param in model.parameters():
    print(type(param), param.size())
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)    # forward
    print(epoch, loss.item())
    
    optimizer.zero_grad()
    loss.backward()                     # backward for grad
    optimizer.step()                    # update

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)

