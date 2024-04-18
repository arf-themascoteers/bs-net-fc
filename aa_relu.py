import matplotlib.pyplot as plt
import numpy as np


def aarelu(y,a):
    x = y.copy()
    x[x<0] = 0
    print(a * np.exp(np.pi/2))
    mask = (a < x) & (x < a * np.exp(np.pi/2))
    x[mask] = a*np.sin(np.log(x[mask]/a)) + a
    x[a * np.exp(np.pi/2)<x] = 2*a
    return x

x = np.linspace(-1,1,10)
y = aarelu(x, 0.3)
plt.plot(x, y)
plt.xlim(-1, 1)
plt.show()
print("\t".join([str(i) for i in x]))
print("\t".join([str(i) for i in y]))