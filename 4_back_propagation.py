
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])     # Tensor是一个类，torch.FloatTensor的别名
w.requires_grad = True

def tensor_test():
    b = w + 1
    c = w.data + 1
    print("b: ", b, b.requires_grad)
    print("c: ", c, c.requires_grad)

# 讲义最后一个练习题
def tensor_grad_test():
    w1 = torch.tensor([1.0], requires_grad=True)    # tensor是一个函数
    w2 = torch.tensor([2.0], requires_grad=True)
    x = 3.0
    b = torch.tensor([4.0], requires_grad=True)
    y = 20.0

    l = (w1*(x**2) + w2*x + b - y)**2
    l.backward()
    print("w1 grad=", w1.grad)
    print("w2 grad=", w2.grad.item())
    print("b grad=", b.grad.item())


def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


if __name__ == "__main__":

    print("predict (before training)", 4, forward(4).item())

    tensor_test()
    tensor_grad_test()

    for epoch in range(10):
        for x, y in zip(x_data, y_data):
            l = loss(x, y)
            l.backward()
            print('\tgrad:', x, y, w.grad.item())   # item()返回具体的数值float
            w.data = w.data - 0.01 * w.grad.data    # data返回一个tensor(requires_grad为False)
            w.grad.data.zero_()
            print("progress:", epoch, l.item())

    print("predict (after training)", 4, forward(4).item())


