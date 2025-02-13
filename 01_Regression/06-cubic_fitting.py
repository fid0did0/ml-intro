import matplotlib.pyplot as plt
import numpy as np
#import cv2 as cv

px=np.array([[ 0.7737313 ], [ 1.01758359], [-0.37149415], [ 1.70712527],
 [-1.48699013], [-2.07644819], [ 2.28298081], [-0.15987864],
 [ 1.7787048 ], [-1.99302454], [-1.32743482], [-2.21539252],
 [-0.81147619], [-1.4379046 ], [-1.01023527], [-1.97638684],
 [ 0.93710512], [ 0.71148664], [-1.20874865], [-0.55489419],
 [-0.65454354], [-1.9441292 ], [ 0.16592805], [ 1.69006497],
 [ 1.89887062]])
py=np.array([[ 0.87261153], [ 1.01150473], [ 1.09993991], [ 1.60523726],
 [-0.76060341], [-3.29933658], [ 3.99128231], [ 0.89114108],
 [ 1.30463954], [-3.09159127], [ 0.25876728], [-3.1132075 ],
 [ 0.80707797], [ 0.12324265], [ 0.42024265], [-2.7434859 ],
 [ 0.92747655], [ 0.89208245], [-0.07419363], [ 0.99353603],
 [ 1.03979391], [-2.31323434], [ 1.08030857], [ 1.99165034],
 [ 2.0520717 ]])
N=px.shape[0]

# plot
fig, ax = plt.subplots()

ax.plot(px, py, 'x', markeredgewidth=2)
a=1.07569217
b=0.50826383
x = np.linspace(-5, 5, 100)
y = a*x+b
ax.plot(x, y)

a=-0.32591243
b=1.04063995
c=1.17420235
y = a*np.power(x,2)+b*x+c
ax.plot(x, y)

ax.set(xlim=(-3.5, 3.5), ylim=(-5, 5))
ax.grid(visible=True)
plt.ion()
plt.show()
input()

A=np.concatenate((np.power(px,3), np.power(px,2), px, np.ones([N,1])), axis=1)
Ainv=np.linalg.pinv(A)
#print(Ainv.shape)

w=np.matmul(Ainv, py)
print(w)
a=w[0]
b=w[1]
c=w[2]
d=w[3]

x = np.linspace(-5, 5, 100)
y = a*np.power(x,3)+b*np.power(x,2)+c*x+d
est_y=a*np.power(px,3)+b*np.power(px,2)+c*px+d
lse=np.sqrt(np.sum((est_y-py)**2))
print('LSE: %0.3f' % lse)

ax.plot(x, y)
plt.show()
plt.pause(10.0)