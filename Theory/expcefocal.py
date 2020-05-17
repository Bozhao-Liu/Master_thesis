import os

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0.01, 0.99, 99)
y_pos = 1/x-1
y_neg = - x**2*np.log(1-x)
fig, ax = plt.subplots()

ax.plot(x, y_pos, label=r'$\frac{1}{\hat{y}_i}-1)$', lw=2)
ax.plot(x, y_neg, label=r'$-(\hat{y}_i)^2*log(1-\hat{y}_i)$', lw=2)

ax.set(xlim=[-0.05, 1.05], ylim = [-1, 30])
ax.set_title('F-EXP_BCE theoretical learning curve', fontsize = 'x-large')
ax.legend(loc='upper right', fontsize = 'x-large')	
ax.set_xlabel('$\hat{y}_i$', fontsize = 'x-large')
ax.set_ylabel('loss', fontsize = 'x-large')
LC_png_file = 'expcefocal.png'
plt.savefig(LC_png_file)

