import os

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0.01, 0.99, 99)
y_pos = - 1/x**2
y_neg = 1/(1-x)
fig, ax = plt.subplots()

ax.plot(x, y_pos, label=r'$-\frac{1}{\hat{y}_i^2}$ (gradient for $(\frac{1}{\hat{y}_i}-1)$)', lw=2)
ax.plot(x, y_neg, label=r'$\frac{1}{1-\hat{y}_i}$ (gradient for $-log(1-\hat{y}_i)$)', lw=2)

ax.set(xlim=[-0.05, 1.05], title='B-EXP_BCE theoretical learning curve gradient')
ax.legend(loc='lower right')	
ax.set_xlabel('$\hat{y}_i$')
ax.set_ylabel('loss')
Grad_png_file = 'expcebcegrad.png'
plt.savefig(Grad_png_file)

