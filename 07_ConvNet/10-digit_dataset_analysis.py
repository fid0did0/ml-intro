import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape)

# Flatten the 28x28 images to 784 features
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)
print(x_train_flat.shape)

# Convert to DataFrames
df_train = pd.DataFrame(x_train_flat)
df_train['label'] = y_train

df_test = pd.DataFrame(x_test_flat)
df_test['label'] = y_test

# Display the first few rows
print(df_train.head())
print(df_train['label'].info())
print(df_train['label'].describe())
print(df_train['label'].value_counts())

# Count of each class
class_counts = df_train['label'].value_counts().sort_index()

# Bar plot
plt.figure(figsize=(8, 5))
plt.bar(class_counts.index, class_counts.values, color='skyblue')
plt.xlabel('Digit Label')
plt.ylabel('Count')
plt.title('MNIST Digit Class Distribution')
plt.xticks(class_counts.index)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

unused=input()

# Create a 2D subplot grid: 1 row, 10 columns
fig, axes = plt.subplots(1, 10, figsize=(15, 3))

# Loop over each digit class (0–9)
for digit in range(10):
    # Get one random sample from the class
    sample = df_train[df_train['label'] == digit].sample(1)
    
    # Extract pixel values and reshape into 28x28
    image = sample.drop('label', axis=1).values.reshape(28, 28)
    
    # Plot it
    axes[digit].imshow(image, cmap='gray')
    axes[digit].set_title(f"Label: {digit}")
    axes[digit].axis('off')

plt.suptitle("One Random Sample per Digit Class", fontsize=16)
plt.tight_layout()
plt.show()
