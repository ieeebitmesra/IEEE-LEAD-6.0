
# Import all modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Load dataset
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training data: {X_train_full.shape}")
print(f"Test data: {X_test.shape}")

# Split training data (45,000 train + 5,000 validation)
X_train = X_train_full[:45000]
y_train = y_train_full[:45000]
X_valid = X_train_full[45000:]
y_valid = y_train_full[45000:]

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_valid.shape}")
print(f"Test set: {X_test.shape}")

# Normalize to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_valid = X_valid.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print("\n✓ Data normalized")

# Data augmentation for better accuracy --- Extra
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

datagen.fit(X_train)
print("✓ Data augmentation configured")

# Build optimized CNN model

def create_best_cnn():
    model = models.Sequential([
        # Block 1
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ])

    return model

model = create_best_cnn()
model.summary()

# Model Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled")

# Set Callbacks
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_cifar10_cnn.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

callbacks = [early_stopping, reduce_lr, checkpoint]

# Train with data augmentation
print("Training started...\n")

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=150,
    validation_data=(X_valid, y_valid),
    callbacks=callbacks,
    verbose=1
)

print("\n Training complete!")

# Visualize training
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], label='Training', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Validation evaluation
val_loss, val_acc = model.evaluate(X_valid, y_valid, verbose=0)
print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

y_valid_pred = np.argmax(model.predict(X_valid, verbose=0), axis=1)
y_valid_true = y_valid.flatten()

print("\n" + "="*50)
print("VALIDATION CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_valid_true, y_valid_pred, target_names=class_names))

# FINAL TEST EVALUATION
print("="*60)
print("FINAL TEST SET EVALUATION")
print("="*60)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\n TEST ACCURACY: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f" TEST LOSS: {test_loss:.4f}")

# Predictions
y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_test_true = y_test.flatten()

# Detailed report
print("\n" + "="*60)
print("TEST SET - CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test_true, y_test_pred, target_names=class_names))