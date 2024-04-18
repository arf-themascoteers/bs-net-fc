import matplotlib.pyplot as plt
import numpy as np


def aarelu(y,a):
    x = y.copy()
    x[x<a] = 0
    return x

x = np.linspace(-2,2,100)
y = aarelu(x, 0.8)
plt.plot(x, y)
plt.xlim(-2, 2)
plt.title("k=0.8")
plt.xlabel("Band Weight (x)")
plt.ylabel("g(x)")
plt.show()
print("\t".join([str(i) for i in x]))
print("\t".join([str(i) for i in y]))