import matplotlib.pyplot as plt
import numpy as np
#import cv2 as cv

N=25
rng = np.random.default_rng()
x=6*(rng.random((N,1))-0.5)
#print(x)
y=0.4*np.power(x,3)-0.3*np.power(x,2)-0.25*x+1
#y=1/(np.power(x,2)+1)
x=x+rng.normal(scale=0.1, size=(N,1))
y=y+rng.normal(scale=0.1, size=(N,1))
print(x)
print(y)

# plot
fig, ax = plt.subplots()

#x = np.linspace(-5, 5, 100)
#y = a*x+b
#ax.plot(x, y)
ax.plot(x, y, 'x', markeredgewidth=2, color='red')
#ax.plot(p1[0], p1[1], 'x', markeredgewidth=2, color='red')

ax.set(xlim=(-3.5, 3.5))
       #ylim=(0, 8)
ax.grid(visible=True)

plt.show()
