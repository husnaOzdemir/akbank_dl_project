import os, pickle, numpy as np, tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

INPUT_DIR = "/kaggle/input/traffic-signs-preprocessed"
PICKLE_NAME = "data8.pickle"  # gerekirse parametreleştirilebilir

def to_channel_last(x):  # (N,C,H,W)->(N,H,W,C)
    return np.transpose(x, (0,2,3,1))

def load_data(pickle_path=os.path.join(INPUT_DIR, PICKLE_NAME)):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    x_train = to_channel_last(data["x_train"]).astype("float32")
    x_val   = to_channel_last(data["x_validation"]).astype("float32")
    x_test  = to_channel_last(data["x_test"]).astype("float32")
    y_train = data["y_train"]; y_val = data["y_validation"]; y_test = data["y_test"]
    lb = LabelBinarizer()
    y_train_oh = lb.fit_transform(y_train)
    y_val_oh   = lb.transform(y_val)
    y_test_oh  = lb.transform(y_test)
    return (x_train, y_train, y_train_oh), (x_val, y_val, y_val_oh), (x_test, y_test, y_test_oh)

def make_augmentation():
    from tensorflow.keras import layers, Sequential
    return Sequential([
        # layers.RandomFlip('horizontal'),  # yön semantiği için devre dışı
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")

def make_ds(X, y_oh, training=False, batch_size=128):
    AUTO = tf.data.AUTOTUNE
    aug = make_augmentation()
    ds = tf.data.Dataset.from_tensor_slices((X, y_oh))
    if training:
        ds = ds.shuffle(4096, reshuffle_each_iteration=True)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTO)
    return ds.batch(batch_size).prefetch(AUTO)

def to_rgb_224(x):
    x = tf.convert_to_tensor(x)
    if x.shape[-1] == 1:
        x = tf.image.grayscale_to_rgb(x)
    x = tf.image.resize(x, (224,224))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
    return x

def make_ds_224(X, y_oh, training=False, batch=64, seed=42):
    AUTO = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((X, y_oh))
    if training:
        ds = ds.shuffle(4096, seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda xx, yy: (to_rgb_224(xx), yy), num_parallel_calls=AUTO)
    return ds.batch(batch).prefetch(AUTO)
