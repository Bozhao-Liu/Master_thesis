import os

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0.00001, 0.99999, 99999)
y_pos = - (1-x)**2*np.log(x)
y_neg = - x**2*np.log(1-x)
fig, ax = plt.subplots()

ax.plot(x, y_pos, label=r'$-(1-\hat{y}_i)^2*log(\hat{y}_i)$', lw=2)
ax.plot(x, y_neg, label=r'$-(\hat{y}_i)^2*log(1-\hat{y}_i)$', lw=2)

ax.set(xlim=[-0.05, 1.05], ylim = [-1, 12])
ax.set_title('Focal Loss theoretical learning curve', fontsize = 'x-large')
ax.legend(loc='upper center', fontsize = 'x-large')	
ax.set_xlabel('$\hat{y}_i$', fontsize = 'x-large')
ax.set_ylabel('loss', fontsize = 'x-large')
LC_png_file = 'FL.png'
plt.savefig(LC_png_file)

