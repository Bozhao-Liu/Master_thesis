import os

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0.01, 0.99, 99)
y_pos = 2*(1-x)*np.log(x)-(1-x)**2/x
y_neg = -2*x*np.log(1-x)+x**2/(1-x)
fig, ax = plt.subplots()

ax.plot(x, y_pos, label=r'$2(1-\hat{y}_i)log(\hat{y}_i)-\frac{(1-\hat{y}_i)^2}{\hat{y}_i}$'
'\n'
'(gradient for $-(1-\hat{y}_i)^2*log(\hat{y}_i)$)', lw=2)
ax.plot(x, y_neg, label=r'$-2\hat{y}_i log(1-\hat{y}_i)+\frac{\hat{y}_i^2}{(1-\hat{y}_i)}$ '
'\n'
'(gradient for $-(\hat{y}_i)^2*log(1-\hat{y}_i)$)', lw=2)

ax.set(xlim=[-0.05, 1.05])
ax.set_title('Focal loss theoretical learning curve gradient', fontsize = 'x-large')
ax.legend(loc='lower right', fontsize = 'x-large')	
ax.set_xlabel('$\hat{y}_i$', fontsize = 'x-large')
ax.set_ylabel('loss', fontsize = 'x-large')
Grad_png_file = 'FLgrad.png'
plt.savefig(Grad_png_file)

