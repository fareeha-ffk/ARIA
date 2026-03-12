import numpy as np
import pandas as pd
import tensorflow as tf

ML_DIR = '/Users/fareeha/ARIA/ml/'

# Load scaler
mean  = np.load(ML_DIR + 'scaler_mean.npy')
scale = np.load(ML_DIR + 'scaler_scale.npy')

# Load model
model = tf.keras.models.load_model(ML_DIR + 'model.h5')

# Extract weights manually
weights_layer1 = model.layers[0].get_weights()  # [W, b]
weights_layer2 = model.layers[2].get_weights()  # [W, b] (layer[1] is dropout)

W1, b1 = weights_layer1  # shape: (5,16) and (16,)
W2, b2 = weights_layer2  # shape: (16,3) and (3,)

print("Layer 1 weights shape:", W1.shape)
print("Layer 2 weights shape:", W2.shape)

# Quantize weights to INT8
def quantize_to_int8(arr):
    max_val = np.max(np.abs(arr))
    scale_factor = 127.0 / max_val
    quantized = np.clip(np.round(arr * scale_factor), -128, 127).astype(np.int8)
    return quantized, scale_factor

W1_q, W1_scale = quantize_to_int8(W1)
W2_q, W2_scale = quantize_to_int8(W2)
b1_q, b1_scale = quantize_to_int8(b1)
b2_q, b2_scale = quantize_to_int8(b2)

# Save quantized weights
np.save(ML_DIR + 'W1_int8.npy', W1_q)
np.save(ML_DIR + 'W2_int8.npy', W2_q)
np.save(ML_DIR + 'b1_int8.npy', b1_q)
np.save(ML_DIR + 'b2_int8.npy', b2_q)

# Save scale factors
scales = {
    'W1_scale': W1_scale,
    'W2_scale': W2_scale,
    'b1_scale': b1_scale,
    'b2_scale': b2_scale
}
np.save(ML_DIR + 'scale_factors.npy', scales)

# Size comparison
original_size = (W1.nbytes + W2.nbytes + b1.nbytes + b2.nbytes)
quantized_size = (W1_q.nbytes + W2_q.nbytes + b1_q.nbytes + b2_q.nbytes)

print(f"\nFloat32 weights size: {original_size} bytes")
print(f"INT8 weights size:    {quantized_size} bytes")
print(f"Size reduction:       {original_size/quantized_size:.1f}x smaller")

# Verify quantized model still works
# Run 100 test samples through both and compare
df = pd.read_csv(ML_DIR + 'dataset.csv')
X  = df[['PM25','VOC','HeatIdx','HR','SpO2']].values
X  = (X - mean) / scale
X_test = X[:100].astype(np.float32)

# Float32 predictions
float_preds = np.argmax(model.predict(X_test, verbose=0), axis=1)

# INT8 manual inference
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

int8_preds = []
for sample in X_test:
    # Layer 1: quantize input
    x_q = np.clip(np.round(sample * 64), -128, 127).astype(np.int8)
    
    # Layer 1 forward pass (dequantize for math)
    h = relu(W1_q.T.astype(np.float32) / W1_scale @ 
             x_q.astype(np.float32) / 64 + 
             b1_q.astype(np.float32) / b1_scale)
    
    # Layer 2 forward pass
    out = W2_q.T.astype(np.float32) / W2_scale @ h + \
          b2_q.astype(np.float32) / b2_scale
    
    int8_preds.append(np.argmax(softmax(out)))

int8_preds = np.array(int8_preds)

# Compare
matches = np.sum(float_preds == int8_preds)
print(f"\nFloat32 vs INT8 agreement: {matches}/100 samples")
print(f"Quantization accuracy loss: {100-matches}%")
print("\nAll quantized weights saved successfully")