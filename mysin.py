import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def my_relu(x, k):
    y = x.clone()
    y = torch.abs(torch.sin(y*torch.pi/2))
    return y

x = torch.linspace(-3,3,1000)
y = my_relu(x, 0.75)
plt.plot(x, y)
plt.xlim(-3, 3)
plt.title("k=0.25")
plt.xlabel("Band Weight (x)")
plt.ylabel("g(x)")
plt.show()
print("\t".join([str(i.item()) for i in x]))
print("\t".join([str(i.item()) for i in y]))