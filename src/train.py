import os, time, json, tensorflow as tf
from tensorflow.keras import callbacks
from utils import set_seeds, save_history
from data import load_data, make_ds, make_ds_224
from model import build_cnn_model, build_tl_model, compile_model

def train(mode="cnn", epochs=30, lr=3e-4, batch=128):
    set_seeds()
    (xtr, ytr, ytr_oh), (xv, yv, yv_oh), (xt, yt, yt_oh) = load_data()
    if mode == "cnn":
        model = build_cnn_model(img_shape=xtr.shape[1:], num_classes=ytr_oh.shape[1])
        model = compile_model(model, lr=lr, optimizer="adam")
        train_ds = make_ds(xtr, ytr_oh, training=True, batch_size=batch)
        val_ds   = make_ds(xv,  yv_oh,  training=False, batch_size=batch)
    else:
        model = build_tl_model(num_classes=ytr_oh.shape[1])
        model = compile_model(model, lr=1e-3, optimizer="adam")
        train_ds = make_ds_224(xtr, ytr_oh, training=True, batch=64)
        val_ds   = make_ds_224(xv,  yv_oh,  training=False, batch=64)

    os.makedirs("models", exist_ok=True); os.makedirs("logs", exist_ok=True)
    ckpt = callbacks.ModelCheckpoint("models/best_model.keras", monitor="val_accuracy",
                                     save_best_only=True, mode="max", verbose=1)
    es   = callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=6,
                                   restore_best_weights=True, verbose=1)
    rlr  = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    log_dir = f"logs/{time.strftime('%Y%m%d-%H%M%S')}"
    tb   = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     callbacks=[ckpt, es, rlr, tb], verbose=1)
    save_history(hist, "history.json")
    print("History saved, best model at models/best_model.keras")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cnn","tl"], default="cnn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    train(mode=args.mode, epochs=args.epochs, lr=args.lr, batch=args.batch)
