import numpy as np
import cv2

# Define color bar properties
bar_height = 100
bar_width = 100
colors = [
    (0, 0, 255),      # Red
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (255, 255, 255),  # White
    (0, 0, 0)         # Black
]

color=colors[3]
box=np.full((bar_height, bar_width, 3), color, dtype=np.uint8)
print(type(box))
print(box.shape)

# Create the color bar image
color_bar = np.hstack([
    np.full((bar_height, bar_width, 3), color, dtype=np.uint8)
    for color in colors
])

# Save the image
cv2.imwrite('color_bar.bmp', color_bar)

# Display the image (optional)
cv2.imshow('Color Bar', color_bar)
cv2.waitKey(0)
cv2.destroyAllWindows()
