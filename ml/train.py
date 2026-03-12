import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ML_DIR = '/Users/fareeha/ARIA/ml/'

# Load data
df = pd.read_csv(ML_DIR + 'dataset.csv')
X = df[['PM25','VOC','HeatIdx','HR','SpO2']].values
y = df['label'].values

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler
np.save(ML_DIR + 'scaler_mean.npy', scaler.mean_)
np.save(ML_DIR + 'scaler_scale.npy', scaler.scale_)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# One-hot encode
y_train_oh = tf.keras.utils.to_categorical(y_train, 3)
y_test_oh  = tf.keras.utils.to_categorical(y_test, 3)

# Model 1: With dropout
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train_oh,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test_oh, verbose=0)
print(f"\nModel WITH dropout — Test accuracy: {acc:.4f}")
model.save(ML_DIR + 'model.h5')

# Model 2: Without dropout
model_nd = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

model_nd.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model_nd.fit(
    X_train, y_train_oh,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

loss_nd, acc_nd = model_nd.evaluate(X_test, y_test_oh, verbose=0)
print(f"Model WITHOUT dropout — Test accuracy: {acc_nd:.4f}")
model_nd.save(ML_DIR + 'model_no_dropout.h5')

print(f"\n--- RESULTS ---")
print(f"Dropout accuracy:    {acc:.4f}")
print(f"No-dropout accuracy: {acc_nd:.4f}")
print("All files saved to", ML_DIR)
# Updated for Week 1 & 2

# Updated for Week 1 & 2 Project
