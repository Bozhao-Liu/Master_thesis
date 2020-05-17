import os

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0.00001, 0.99999, 99999)
y_pos = - 1/x**2
y_neg = -2*x*np.log(1-x)+x**2/(1-x)
fig, ax = plt.subplots()

ax.plot(x, y_pos, label=r'$-\frac{1}{\hat{y}_i^2}$ (gradient for $(\frac{1}{\hat{y}_i}-1)$)', lw=2)
ax.plot(x, y_neg, label=r'$-2\hat{y}_i log(1-\hat{y}_i)+\frac{\hat{y}_i^2}{(1-\hat{y}_i)}$ '
			"\n"
			r'(gradient for $-(\hat{y}_i)^2*log(1-\hat{y}_i)$)', lw=2)

ax.set(xlim=[-0.05, 1.05], ylim= [-100, 60])
ax.set_title('F-EXP_BCE theoretical learning curve gradient', fontsize = 'x-large')
ax.legend(loc='lower right', fontsize = 'x-large')	
ax.set_xlabel('$\hat{y}_i$', fontsize = 'x-large')
ax.set_ylabel('loss', fontsize = 'x-large')
Grad_png_file = 'expcefocagrad.png'
plt.savefig(Grad_png_file)

