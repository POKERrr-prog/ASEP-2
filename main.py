import os
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Constants
IMG_SIZE = 64
BATCH_SIZE = 32
DATA_DIR = "asl_alphabet_train"

# Class mapping
all_classes = sorted(os.listdir(DATA_DIR))
class_to_label = {cls: idx for idx, cls in enumerate(all_classes)}
label_to_class = {idx: cls for cls, idx in class_to_label.items()}

# Save label map
with open("asl_labels.json", "w") as f:
    json.dump(label_to_class, f)

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    BatchNormalization(),
    Dropout(0.3),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(all_classes), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("asl_model_best.keras", monitor='val_loss', save_best_only=True)

# Train
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=60,
    callbacks=[early_stop, checkpoint]
)

# Evaluate
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"✅ Validation Accuracy: {val_accuracy * 100:.2f}%")

# Save final model
model.save("asl_model_final.keras")