import os

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0.01, 0.99, 99)
y_pos = - (1-x)**2*np.log(x)
y_neg = - x**2*np.log(1-x)
fig, ax = plt.subplots()

ax.plot(x, y_pos, label=r'$-(1-\hat{y}_i)^2*log(\hat{y}_i)$', lw=2)
ax.plot(x, y_neg, label=r'$-(\hat{y}_i)^2*log(1-\hat{y}_i)$', lw=2)

ax.set(xlim=[-0.05, 1.05], title='Focal Loss theoretical learning curve')
ax.legend(loc='upper center')	
ax.set_xlabel('$\hat{y}_i$')
ax.set_ylabel('loss')
LC_png_file = 'FL.png'
plt.savefig(LC_png_file)

