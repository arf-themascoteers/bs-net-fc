import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def my_relu(x, k):
    y = x.clone()
    y[(y >= -k) & (y <= k)] = 0
    return y

x = torch.linspace(-1,1,100)
y = my_relu(x, 0.25)
plt.plot(x, y)
plt.xlim(-1, 1)
plt.title("k=0.25")
plt.xlabel("Band Weight (x)")
plt.ylabel("g(x)")
plt.show()
print("\t".join([str(i) for i in x]))
print("\t".join([str(i) for i in y]))