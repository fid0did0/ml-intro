import matplotlib.pyplot as plt
import numpy as np

def relu(x):
    rl=np.maximum(x,0)
    return rl


x_test=np.linspace(-3.0, 3.0, 1024)
y_true=(x_test**2-2*x_test+3)

w0=np.array([-0.98551774, -1.5313786 ,  1.2792082 ,  0.61568356,  1.652143  ])
b0=np.array([1.6639476, 0.6620356, 2.296268 , 2.5549433, 1.1802405])
w1=np.array([[ 2.6059532],[ 1.5763674],[ 1.7155551],[-2.8108342],[ 1.3292016]])
b1=-0.622404

y_c=w1[0]*relu(w0[0]*x_test+b0[0])
y_c=np.column_stack((y_c, w1[1]*relu(w0[1]*x_test+b0[1])))
y_c=np.column_stack((y_c, w1[2]*relu(w0[2]*x_test+b0[2])))
y_c=np.column_stack((y_c, w1[3]*relu(w0[3]*x_test+b0[3])))
y_c=np.column_stack((y_c, w1[4]*relu(w0[4]*x_test+b0[4])))
y_test=y_c[:,0]+y_c[:,1]+y_c[:,2]+y_c[:,3]+y_c[:,4]+b1

print(x_test.shape)
# plot
ax1 = plt.subplot(2,1,1)
ax1.plot(x_test, y_true, color='red')
ax2 = plt.subplot(2,1,2)
ax2.plot(x_test, y_c)
ax1.plot(x_test, y_test)

ax1.grid(visible=True)
ax2.grid(visible=True)

plt.show()


