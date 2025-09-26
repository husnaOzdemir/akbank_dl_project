from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers

def build_cnn_model(img_shape, num_classes,
                    base_filters=32, dropout_rate=0.4, l2_reg=1e-4,
                    kernel_size=3, dense_units=256):
    reg = regularizers.l2(l2_reg)
    inputs = keras.Input(shape=img_shape)
    x = inputs
    for mult in [1,2]:
        x = layers.Conv2D(base_filters*mult, kernel_size, padding="same", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
        x = layers.Conv2D(base_filters*mult, kernel_size, padding="same", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x); x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(base_filters*4, kernel_size, padding="same", kernel_regularizer=reg, name="last_conv")(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x); x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="cnn_gtsrb")

def build_tl_model(num_classes):
    inputs = keras.Input(shape=(224,224,3))
    base = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(224,224,3))
    base.trainable = False
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="tl_mnetv2")

def compile_model(model, lr=3e-4, optimizer="adam"):
    if optimizer == "adam":
        opt = optimizers.Adam(lr)
    else:
        opt = optimizers.SGD(lr, momentum=0.9)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
