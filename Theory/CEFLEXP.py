import os

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0.01, 0.3, 99)
y_BCE = - np.log(x)
y_FL = - (1-x)**2*np.log(x)
y_pos = 1/x-1
fig, ax = plt.subplots()

ax.plot(x, y_BCE, label=r'BCE ($-log(\hat{y}_i)$)', lw=2)
ax.plot(x, y_FL, label=r'Focal ($-(1-\hat{y}_i)^2*log(\hat{y}_i)$)', lw=2)
ax.plot(x, y_pos, label=r'EXP_BCE ($\frac{1}{\hat{y}_i}-1$)', lw=2)

ax.set(xlim=[-0.05, 0.4], ylim=[-1,20])
ax.set_title('BCE, Focal, EXP_BCE loss comparison 0 -- 0.3', fontsize = 'x-large')
ax.legend(loc='upper right', fontsize = 'x-large')	
ax.set_xlabel('$\hat{y}_i$', fontsize = 'x-large')
ax.set_ylabel('loss', fontsize = 'x-large')
LC_png_file = 'CEFLEXP0-3.PNG'
plt.savefig(LC_png_file)

