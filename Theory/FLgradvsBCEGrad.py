import os

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0.01, 0.99, 99)
y_BCE = - 1/x
y_FL = 2*(1-x)*np.log(x)-(1-x)**2/x
fig, ax = plt.subplots()

ax.plot(x, y_BCE, label=r'BCE gradient ($-\frac{1}{\hat{y}_i}$)', lw=2)
ax.plot(x, y_FL, label=r'Focal loss gradient ($2(1-\hat{y}_i)log(\hat{y}_i)-\frac{(1-\hat{y}_i)^2}{\hat{y}_i}$)', lw=2)

ax.set(xlim=[-0.05, 1.05], title='Focal loss, BCE gradient Comparison')
ax.legend(loc='lower right')	
ax.set_xlabel('$\hat{y}_i$')
ax.set_ylabel('loss')
Grad_png_file = 'FLgradvsBCEgrad.png'
plt.savefig(Grad_png_file)

