import os

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0.01, 0.99, 99)
y_BCE = - np.log(x)
y_FL = - (1-x)**2*np.log(x)
fig, ax = plt.subplots()

ax.plot(x, y_BCE, label=r'$-log(\hat{y}_i)$', lw=2)
ax.plot(x, y_FL, label=r'$-(1-\hat{y}_i)^2*log(\hat{y}_i)$', lw=2)

ax.set(xlim=[-0.05, 1.05], title='BCE, Focal loss comparison')
ax.legend(loc='upper right')	
ax.set_xlabel('$\hat{y}_i$')
ax.set_ylabel('loss')
LC_png_file = 'FLCE.PNG'
plt.savefig(LC_png_file)

