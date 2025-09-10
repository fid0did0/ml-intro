import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

def from_one_hot(one_hot):
    """
    Converts a one-hot encoded array back to scalar class labels.

    Parameters:
        one_hot (np.ndarray): A 2D numpy array of shape (N, num_classes)
                              where each row is a one-hot encoded vector.

    Returns:
        np.ndarray: A 1D array of scalar labels.
    """
    return np.argmax(one_hot, axis=1)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#print(x_train.shape)

## Filter for digits 0 and 1 in training set
#train_filter = (y_train == 0) | (y_train == 1)
#x_train_01 = x_train[train_filter].astype('float32')/256.0
#y_train_01 = y_train[train_filter].astype('float32')

test_filter = (y_test == 0) | (y_test == 1)
x_test_01 = x_test[test_filter].astype('float32')/256.0
y_test_01_true = y_test[test_filter].astype('int')

conv_model=tf.keras.models.load_model('zero_one_oh_classifier.keras')
conv_model.summary()

print(x_test_01.shape)
y_test_01_pred = conv_model.predict(x_test_01)
y_test_01_pred_scalar = from_one_hot(y_test_01_pred)
print(y_test_01_pred.shape)
print(y_test_01_pred_scalar.shape)

idx=5
print(y_test_01_pred[idx,:])
print(y_test_01_pred_scalar[idx])

error_idx=[]
cm=np.zeros((2, 2), dtype='int')
for k in range(len(y_test_01_pred_scalar)):
    cm[y_test_01_true[k], y_test_01_pred_scalar[k]] += 1
    if y_test_01_true[k]!=y_test_01_pred_scalar[k]:
        error_idx.append(k)

print(cm)

fig, ax = plt.subplots(figsize = (4, 4))
im = ax.imshow(np.log(cm+1.0), cmap='coolwarm')

ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_xticks(range(2))
ax.set_yticks(range(2))
# Loop over data dimensions and create text annotations.
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, cm[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Confusion Matrix")
plt.show()

if len(error_idx):
    fig, axes = plt.subplots(1, len(error_idx), figsize=(15, 3))
    if len(error_idx)>1:
        for i in range(len(error_idx)):
            axes[i].imshow(x_test_01[error_idx[i],:,:], cmap='gray')
            axes[i].set_title(f"Label: {y_test_01_true[error_idx[i]]}, Pred: {y_test_01_pred_scalar[error_idx[i]]}")
            axes[i].axis('off')
    else:
        axes.imshow(x_test_01[error_idx[0],:,:], cmap='gray')
        axes.set_title(f"Label: {y_test_01_true[error_idx[0]]}, Pred: {y_test_01_pred_scalar[error_idx[0]]}")
        axes.axis('off')
    plt.show()

filename = 'Test00_sc.png'
#filename = 'Test01_sc.png'
#filename = 'Test02_sc.png'
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.bitwise_not(img)
img = img/img.max()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#img = cv2.dilate(img, kernel, iterations=1)
#https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
#img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
img = 2*img
img = np.minimum(img, 1.0)

print(img.shape)
img_rs = img.reshape(1, 28, 28)
print(img_rs.shape)
predict_value = conv_model.predict(img_rs)
digit = from_one_hot(predict_value)
f_predict_value = np.array2string(predict_value, formatter={'float_kind':lambda x: f"{x:0.3f}"})
print(f"predict_value = {predict_value}")
print(f"digit = {digit}")

fig, ax = plt.subplots()
ax.clear()
ax.imshow(img_rs[0,:,:], cmap='gray')
ax.set_title(f"Label: 1, Pred: {digit[0]}")
ax.axis('off')

plt.show()