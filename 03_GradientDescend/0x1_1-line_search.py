import matplotlib.pyplot as plt
import numpy as np
#import cv2 as cv

x = np.linspace(-10, 10, 100)
y = -1/(1+0.6*np.power(x,2))
yd1=np.divide(x, 0.3*np.power(x,4)+np.power(x,2)+0.8333)


xk=np.array([-8]);
yk=-1/(1+0.6*np.power(xk,2))
ykd1=np.divide(xk, 0.3*np.power(xk,4)+np.power(xk,2)+0.8333)
sdef=5.6
s_step=5
smin=sdef/(2^s_step)
x_history=np.zeros((3,s_step))
y_history=np.zeros((3,s_step))
s=sdef
xt0=xk[-1]
yt0=-1/(1+0.6*np.power(xt0,2))
ykd1=np.divide(xt0, 0.3*np.power(xt0,4)+np.power(xt0,2)+0.8333)
for kk in range(3):
       k=0
       x_history[kk, k]=xt0
       y_history[kk, k]=yt0
       if (ykd1>0):
              xt=xt0-s
              yt=-1/(1+0.6*np.power(xt,2))
       else:
              xt=xt0+s
              yt=-1/(1+0.6*np.power(xt,2))
       k=1
       x_history[kk, k]=xt
       y_history[kk, k]=yt
       print(y_history.shape)
       print("xt0=%0.3f yt0=%0.3f x=%0.3f y=%0.3f" % (xt0, yt0, xt, yt))
       while((yt>yt0) & (s>smin)):
              s=s/2
              if (ykd1>0):
                     xt=xt0-s
                     yt=-1/(1+0.6*np.power(xt,2))
              else:
                     xt=xt0+s
                     yt=-1/(1+0.6*np.power(xt,2))
              k=k+1
              x_history[kk, k]=xt
              y_history[kk, k]=yt
              print(y_history.shape)
              print("xt0=%0.3f yt0=%0.3f x=%0.3f y=%0.3f" % (xt0, yt0, xt, yt))
       xk=np.append(xk, xt);
       yk=np.append(yk, yt);
       s=sdef
       xt0=xk[-1]
       xt=xt0
       yt0=-1/(1+0.6*np.power(xt,2))
       yt=yt+1
       ykd1=np.divide(xt, 0.3*np.power(xt,4)+np.power(xt,2)+0.8333)
print(x_history)
print(y_history)

#plot
fig, ax = plt.subplots()

ax.plot(x, y)
for kk in range(3):
       ax.plot((x_history[kk,0], x_history[kk,1]), (y_history[kk,0], y_history[kk,0]), color='black', ls='--')
       
step=0
ax.plot((x_history[step,1], x_history[step,1]), (y_history[step,0], y_history[step,1]), color='green', marker='.')
step=1
ax.plot((x_history[step,1], x_history[step,1]), (y_history[step,0], y_history[step,1]), color='red', marker='.')
ax.plot((x_history[step,2], x_history[step,2]), (y_history[step,0], y_history[step,2]), color='green', marker='.')
step=2
ax.plot((x_history[step,1], x_history[step,1]), (y_history[step,0], y_history[step,1]), color='red', marker='.')
ax.plot((x_history[step,2], x_history[step,2]), (y_history[step,0], y_history[step,2]), color='red', marker='.')
ax.plot((x_history[step,3], x_history[step,3]), (y_history[step,0], y_history[step,3]), color='red', marker='.')
ax.plot((x_history[step,4], x_history[step,4]), (y_history[step,0], y_history[step,4]), color='green', marker='.')

#ax.plot(xk, yk, '-x', color='red')
##ax.plot(x, yd1)
##ax.plot(p0[0], p0[1], 'x', markeredgewidth=2, color='red')

#ax.set(xlim=(-10, 10))
#       #ylim=(0, 8)
#ax.grid(visible=True)

plt.ion()
plt.show()
plt.pause(10.0)