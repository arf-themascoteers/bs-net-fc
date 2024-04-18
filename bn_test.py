import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bn1 = nn.BatchNorm1d(4)

    def forward(self, x):
        x = self.bn1(x)
        return x


net = Net()
input_data = torch.randn(3, 4)
print(input_data)
output = net(input_data)
print(output)
