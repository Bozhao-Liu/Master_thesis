import os

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0.01, 0.99, 99)
y_pos = - np.log(x)
y_neg = - np.log(1-x)
fig, ax = plt.subplots()

ax.plot(x, y_pos, label=r'$-log(\hat{y}_i)$', lw=2)
ax.plot(x, y_neg, label=r'$-log(1-\hat{y}_i)$', lw=2)

ax.set(xlim=[-0.05, 1.05], title='BCE theoretical learning curve')
ax.legend(loc='upper center')	
ax.set_xlabel('$\hat{y}_i$')
ax.set_ylabel('loss')
LC_png_file = 'BCE.png'
plt.savefig(LC_png_file)

