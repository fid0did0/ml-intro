import numpy as np
import cv2

# Image dimensions
width = 256  # Width of the color bar
height = 50  # Height of the color bar

# Create a horizontal gradient from 0 (black) to 255 (white)
gradient = np.tile(np.arange(0, 256, dtype=np.uint8), (height, 1))

# Save the image
cv2.imwrite('grayscale_color_bar.bmp', gradient)

# Display the image (optional)
cv2.imshow('Grayscale Color Bar', gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()
