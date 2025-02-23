import matplotlib.pyplot as plt
import numpy as np
#import cv2 as cv

def ffail(x):
       condlist = [x<-1, x>1]
       choicelist = [3*np.power((1+x),2)/4-2*(1+x), 3*np.power((1-x),2)/4-2*(1-x)]
       return(np.select(condlist, choicelist, np.power(x,2)-1))

def ffail_prime(x):
       condlist = [x<-1, x>1]
       choicelist = [(3*x-1)/2, (3*x+1)/2]
       return(np.select(condlist, choicelist, 2*x))

x = np.linspace(-3, 3, 500)
y = ffail(x)
yd1=ffail_prime(x)


xk=np.array([-2.5]);
yk=ffail(xk)
ykd1=np.divide(xk, 0.3*np.power(xk,4)+np.power(xk,2)+0.8333)

cnt=0
sdef=1
s=sdef
xt0=xk[-1]
xt=xt0
yt0=ffail(xt)
yt=yt0+1
ykd1=ffail_prime(xt)
while((np.abs(ykd1)>0.001) & (cnt<10)):
       while((yt>yt0) & (s!=0.1)):
              if (ykd1>0):
                     xt=xt0-s
                     yt=ffail(xt)
              else:
                     xt=xt0+s
                     yt=ffail(xt)
              #print("%0.2f %0.2f %0.2f %0.2f " % (s, xt, yt0, yt))
              s=s/2
       xk=np.append(xk, xt)
       yk=np.append(yk, yt)
       s=sdef
       xt0=xk[-1]
       xt=xt0
       yt0=ffail(xt)
       yt=yt+1
       ykd1=ffail_prime(xt)
       cnt=cnt+1
print(yk)
print(xk[-1])
print(yk[-1])

# plot
fig, ax = plt.subplots()

ax.plot(x, y)
ax.plot(xk, yk, '-x', color='red')
#ax.plot(x, yd1)
#ax.plot(p0[0], p0[1], 'x', markeredgewidth=2, color='red')

ax.set(xlim=(-3, 3))
#ax.set(xlim=(-10, 10))
       #ylim=(-2, 1))
ax.grid(visible=True)

plt.ion()
plt.show()
plt.pause(10.0)