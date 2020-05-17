import os

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0.01, 0.99, 99)
y_BCE = - 1/x
y_FL = 2*(1-x)*np.log(x)-(1-x)**2/x
y_pos = - 1/x**2
fig, ax = plt.subplots()

ax.plot(x, y_BCE, label=r'BCE Gradient ($-\frac{1}{\hat{y}_i}$)', lw=2)
ax.plot(x, y_FL, label=r'Focal Gradient ($2(1-\hat{y}_i)log(\hat{y}_i)-\frac{(1-\hat{y}_i)^2}{\hat{y}_i}$)', lw=2)
ax.plot(x, y_pos, label=r'EXP_BCE Gradient ($-\frac{1}{\hat{y}_i^2}$)', lw=2)

ax.set(xlim=[-0.05, 1.05], ylim = [-300,1])
ax.set_title('BCE, Focal, EXP_BCE loss gradient comparison 0 -- 1', fontsize = 'x-large')
ax.legend(loc='lower right', fontsize = 'x-large')	
ax.set_xlabel('$\hat{y}_i$', fontsize = 'x-large')
ax.set_ylabel('loss', fontsize = 'x-large')
LC_png_file = 'CEFLEXPgrad0-1.PNG'
plt.savefig(LC_png_file)

