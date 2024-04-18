import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

def my_relu(x, k):
    y = x.clone()
    y[y<k] = torch.exp(-1*(0.25 - y[y<k] )) * 0.25 / torch.exp(-1*(0.25 - y[y<k] ))
    return y

x = torch.linspace(-2,2,100)
y = my_relu(x, 0.8)
plt.plot(x, y)
plt.xlim(-2, 2)
plt.title("k=0.8")
plt.xlabel("Band Weight (x)")
plt.ylabel("g(x)")
plt.show()
print("\t".join([str(i) for i in x]))
print("\t".join([str(i) for i in y]))