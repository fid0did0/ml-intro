import matplotlib.pyplot as plt
import numpy as np
#import cv2 as cv

x = np.linspace(-10, 10, 100)
y = -1/(1+0.6*np.power(x,2))
yd1=np.divide(x, 0.3*np.power(x,4)+np.power(x,2)+0.8333)


xk=np.array([-8]);
yk=-1/(1+0.6*np.power(xk,2))
ykd1=np.divide(xk, 0.3*np.power(xk,4)+np.power(xk,2)+0.8333)

sdef=3
s=sdef
xt0=xk[-1]
xt=xt0
yt0=-1/(1+0.6*np.power(xt,2))
yt=yt0+1
ykd1=np.divide(xt, 0.3*np.power(xt,4)+np.power(xt,2)+0.8333)
while(np.abs(ykd1)>0.001):
       while((yt>yt0) & (s!=0.1)):
              if (ykd1>0):
                     xt=xt0-s
                     yt=-1/(1+0.6*np.power(xt,2))
              else:
                     xt=xt0+s
                     yt=-1/(1+0.6*np.power(xt,2))
              s=s/2
       xk=np.append(xk, xt);
       yk=np.append(yk, yt);
       s=sdef
       xt0=xk[-1]
       xt=xt0
       yt0=-1/(1+0.6*np.power(xt,2))
       yt=yt+1
       ykd1=np.divide(xt, 0.3*np.power(xt,4)+np.power(xt,2)+0.8333)
print(xk[-1])
print(yk[-1])

# plot
fig, ax = plt.subplots()

ax.plot(x, y)
ax.plot(xk, yk, '-x', color='red')
#ax.plot(x, yd1)
#ax.plot(p0[0], p0[1], 'x', markeredgewidth=2, color='red')

ax.set(xlim=(-10, 10))
       #ylim=(0, 8)
ax.grid(visible=True)

plt.ion()
plt.show()
plt.pause(10.0)