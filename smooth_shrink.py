import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def my_relu(x, k):
    y = x.clone()
    mask = (y >= -k) & (y <= k)
    #y[mask] = torch.sin(y[mask]*torch.pi/16) / torch.sin(torch.tensor(k*torch.pi/16)) * y[mask] * (y[mask]/torch.abs(y[mask]))
    #n = 0.1
    #y[mask] = torch.sin(y[mask]*n) / torch.sin(torch.tensor(k*n)) * y[mask] * (y[mask]/torch.abs(y[mask]))
    n = 0.3
    y[mask] = (torch.exp(y[mask]*n)-torch.exp(-y[mask]*n)) / (torch.exp(y[mask]*n)+torch.exp(-y[mask]*n))
    return y

x = torch.linspace(-2,2,1000)
y = my_relu(x, 0.8)
plt.plot(x, y)
plt.xlim(-2, 2)
plt.title("k=0.8")
plt.xlabel("Band Weight (x)")
plt.ylabel("g(x)")
plt.show()
print("\t".join([str(i.item()) for i in x]))
print("\t".join([str(i.item()) for i in y]))