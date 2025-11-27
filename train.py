import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = 'data/keypoint_normalized.csv'
MODEL_PATH = 'model/victory.tflite'
HDF5_PATH = 'model/victory.hdf5'
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Load data
X_dataset = np.loadtxt(DATA_PATH, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
y_dataset = np.loadtxt(DATA_PATH, delimiter=',', dtype='int32', usecols=(0))

logger.info(f"Raw keypoints: min={X_dataset.min():.3f}, max={X_dataset.max():.3f}")
logger.info(f"Class distribution: {np.bincount(y_dataset)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_dataset, y_dataset, train_size=0.75, random_state=42
)

# Model architecture (exact Kinivs match)
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2, )),  # (42,)
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

logger.info(f"Model input shape: {model.input_shape}")
logger.info(f"Model output shape: {model.output_shape}")

# Callbacks
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    HDF5_PATH, verbose=1, save_weights_only=False
)
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

# Train
model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback]
)

# Evaluate
train_loss, train_accuracy = model.evaluate(X_train, y_train, batch_size=128)
logger.info(f"Train accuracy: {train_accuracy:.4f}")
test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=128)
logger.info(f"Test accuracy: {test_accuracy:.4f}")

# Save Keras model
model.save(HDF5_PATH, include_optimizer=False)
logger.info(f"Saved Keras model to {HDF5_PATH}")

# Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(MODEL_PATH, 'wb') as f:
    f.write(tflite_model)
logger.info(f"Saved TFLite model to {MODEL_PATH}")