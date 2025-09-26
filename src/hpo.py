import os, gc, itertools, numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import callbacks, optimizers
from data import load_data, make_ds
from model import build_cnn_model, compile_model

class TimeLimit(callbacks.Callback):
    def __init__(self, seconds=300): super().__init__(); self.s=seconds; self.t=None
    def on_train_begin(self, logs=None): import time; self.t=time.time()
    def on_train_batch_end(self, batch, logs=None):
        import time; 
        if time.time()-self.t>self.s: self.model.stop_training=True

def main():
    (xtr, ytr, ytr_oh), (xv, yv, yv_oh), _ = load_data()
    AUTO = tf.data.AUTOTUNE
    BATCH = 128
    train_ds = make_ds(xtr, ytr_oh, training=True, batch_size=BATCH).take(100).cache().prefetch(AUTO)
    val_ds   = make_ds(xv,  yv_oh,  training=False, batch_size=BATCH).take(40).cache().prefetch(AUTO)
    opt = tf.data.Options(); opt.experimental_deterministic = False
    train_ds = train_ds.with_options(opt); val_ds = val_ds.with_options(opt)

    search = {
        "base_filters":[32,48],
        "dropout":[0.3,0.5],
        "lr":[1e-3,3e-4],
        "kernel":[3,5],
        "dense":[128,256],
        "batch":[64,128],
        "opt":["adam","sgd"]
    }
    results=[]
    for bf,dr,lr,ks,du,bs,op in itertools.product(
        search["base_filters"], search["dropout"], search["lr"], search["kernel"],
        search["dense"], search["batch"], search["opt"]
    ):
        tf.keras.backend.clear_session(); gc.collect()
        m = build_cnn_model(img_shape=xtr.shape[1:], num_classes=ytr_oh.shape[1],
                            base_filters=bf, dropout_rate=dr, kernel_size=ks, dense_units=du)
        m = compile_model(m, lr=lr, optimizer=op)
        h = m.fit(
            train_ds.unbatch().batch(bs).take(100),
            validation_data=val_ds.unbatch().batch(bs).take(40),
            epochs=8, verbose=0,
            callbacks=[callbacks.EarlyStopping("val_accuracy", patience=2, restore_best_weights=True),
                       callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=1),
                       TimeLimit(300)]
        )
        val_acc = float(max(h.history.get("val_accuracy",[0.0])))
        results.append({"bf":bf,"dr":dr,"lr":lr,"ks":ks,"du":du,"bs":bs,"opt":op,"val_acc":val_acc})
    df = pd.DataFrame(results).sort_values("val_acc", ascending=False)
    df.to_csv("hp_results_extended.csv", index=False); print(df.head(10))

if __name__ == "__main__":
    main()
