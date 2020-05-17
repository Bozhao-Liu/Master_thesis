import os

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0.01, 0.99, 99)
y_pos = - 1/x**2
y_neg = 1/(1-x)**2
fig, ax = plt.subplots()

ax.plot(x, y_pos, label=r'$-\frac{1}{\hat{y}_i^2}$ (gradient for $(\frac{1}{\hat{y}_i}-1)$)', lw=2)
ax.plot(x, y_neg, label=r'$\frac{1}{(1-\hat{y}_i)^2}$ (gradient for $(\frac{1}{1-\hat{y}_i}-1)$)', lw=2)

ax.set(xlim=[-0.05, 1.05], ylim = [-100, 100])
ax.set_title('EXP_BCE theoretical learning curve gradient', fontsize = 'x-large')
ax.legend(loc='lower right', fontsize = 'x-large')	
ax.set_xlabel('$\hat{y}_i$', fontsize = 'x-large')
ax.set_ylabel('loss', fontsize = 'x-large')
Grad_png_file = 'expcegrad.png'
plt.savefig(Grad_png_file)

