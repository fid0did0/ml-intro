import matplotlib.pyplot as plt
import numpy as np

# Create a gradient image for the color bar
gradient_1d = np.linspace(0, 1, 256)
print(type(gradient_1d))
print(gradient_1d.shape)
gradient = np.vstack((gradient_1d.reshape(1, -1), gradient_1d.reshape(1, -1)))
print(gradient.shape)

# Create the figure and display the gradient
fig, ax = plt.subplots(2,1,figsize=(6, 4))
#ax.set_title('Grayscale Color Bar')
ax[0].plot(gradient_1d, gradient_1d)

ax[1].imshow(gradient, aspect='auto', cmap='gray')
ax[1].set_axis_off()

# Save the figure as an image
#plt.savefig('grayscale_color_bar.png', bbox_inches='tight', pad_inches=0.1)
plt.show()
