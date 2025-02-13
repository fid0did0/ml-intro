import matplotlib.pyplot as plt
import numpy as np
#import cv2 as cv

p0=np.array([4, 7])
p1=np.array([-2,-1])

x=np.array([[p0[0]],[p1[0]]])
y=np.array([[p0[1]],[p1[1]]])
#print(x)

A=np.concatenate((x, np.ones([2,1])), axis=1)
Ainv=np.linalg.inv(A)
#print(A)

w=np.matmul(Ainv, y)
#print(w)
a=w[0]
b=w[1]
#print(a)

# plot
fig, ax = plt.subplots()

x = np.linspace(-5, 5, 100)
y = a*x+b
ax.plot(x, y)
ax.plot(p0[0], p0[1], 'x', markeredgewidth=2, color='red')
ax.plot(p1[0], p1[1], 'x', markeredgewidth=2, color='red')

ax.set(xlim=(-5, 5))
       #ylim=(0, 8)
ax.grid(visible=True)

plt.show()
