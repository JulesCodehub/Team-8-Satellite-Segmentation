import os
import numpy as np
from glob import glob
import tifffile
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from unet_model import unet_model

def load_data(img_dir, mask_dir, num_classes=5):
    X, Y = [], []
    for img_path in sorted(glob(os.path.join(img_dir, '*.tif'))):
        mask_path = os.path.join(mask_dir, os.path.basename(img_path))
        if os.path.exists(mask_path):
            img = tifffile.imread(img_path)
            mask = tifffile.imread(mask_path)

            X.append(img)
            Y.append(to_categorical(mask, num_classes=num_classes))

    return np.array(X), np.array(Y)

img_dir = './processed/images'
mask_dir = './processed/masks'
X, Y = load_data(img_dir, mask_dir)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=42)

model = unet_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=25, batch_size=16)
