"""
model/model_builder.py

CRITICAL FIX:
  Old code used:  model.add(Conv2D(..., input_shape=input_shape))
  This causes TF 2.16+ to write 'batch_shape' in the saved config
  which NOTHING can reload correctly.

  New code uses:  keras.Input(shape=...) as a separate first layer
  This writes 'batch_input_shape' which loads correctly everywhere.
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense, Dropout
)


def build_model(input_shape=(128, 128, 3), num_classes=2) -> Model:
    """
    Build CNN using Functional API with explicit Input layer.

    Why Functional API instead of Sequential + input_shape arg?
    Sequential's input_shape shortcut causes TF 2.16+ to save
    'batch_shape' in config — which breaks model loading.
    Explicit Input() layer saves 'batch_input_shape' — universally compatible.
    """
    inputs = Input(shape=input_shape, name='input_layer')

    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = MaxPooling2D((2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = MaxPooling2D((2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = MaxPooling2D((2, 2), name='pool3')(x)

    # Block 4 — extra depth for better feature learning
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4')(x)
    x = BatchNormalization(name='bn4')(x)
    x = MaxPooling2D((2, 2), name='pool4')(x)

    # Classifier head
    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu', name='dense1')(x)
    x = Dropout(0.5, name='dropout1')(x)
    x = Dense(128, activation='relu', name='dense2')(x)
    x = Dropout(0.3, name='dropout2')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='NeuroSnapCNN')
    return model


if __name__ == '__main__':
    m = build_model()
    m.summary()
    print("\nOutput shape:", m.output_shape)
    print("Input  shape:", m.input_shape)
