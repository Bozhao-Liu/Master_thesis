import os

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0.01, 0.99, 99)
y_pos = 1/x-1
y_neg = 1/(1-x)-1
fig, ax = plt.subplots()

ax.plot(x, y_pos, label=r'$\frac{1}{\hat{y}_i}-1)$', lw=2)
ax.plot(x, y_neg, label=r'$(\frac{1}{1-\hat{y}_i}-1)$', lw=2)

ax.set(xlim=[-0.05, 1.05], title='EXP_BCE theoretical learning curve')
ax.legend(loc='upper center')	
ax.set_xlabel('$\hat{y}_i$')
ax.set_ylabel('loss')
LC_png_file = 'expce.png'
plt.savefig(LC_png_file)

