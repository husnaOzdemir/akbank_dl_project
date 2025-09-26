import os, numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from data import load_data, make_ds
os.makedirs("reports/figures", exist_ok=True)

def get_last_conv_layer_name(m):
    for layer in reversed(m.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
    raise ValueError("Conv2D yok.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, target_size=None):
    grad_model = keras.models.Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_out = conv_out[0]
    heatmap = tf.tensordot(conv_out, pooled_grads, axes=(2,0))
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)
    if target_size:
        H,W = target_size
        heatmap = tf.image.resize(heatmap[...,None], (H,W)); heatmap = tf.squeeze(heatmap, -1)
    return heatmap.numpy()

def main(model_path="models/best_model.keras", num=5):
    (_,_,_), (_,_,_), (xt, yt, yt_oh) = load_data()
    model = tf.keras.models.load_model(model_path)
    last_conv = get_last_conv_layer_name(model)
    sample = xt[:num]
    preds = model.predict(sample, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    plt.figure(figsize=(12,6))
    for i in range(num):
        img = sample[i:i+1]
        H,W = sample[i].shape[:2]
        heatmap = make_gradcam_heatmap(img, model, last_conv, target_size=(H,W))
        # Görüntüle
        plt.subplot(2, num, i+1)
        disp = sample[i].squeeze()
        if disp.ndim==2: plt.imshow(disp, cmap="gray")
        else: plt.imshow(disp)
        plt.axis("off"); plt.title(f"Pred: {pred_labels[i]}")
        plt.subplot(2, num, num+i+1)
        plt.imshow(disp if disp.ndim!=2 else disp, cmap="gray")
        plt.imshow(heatmap, alpha=0.4, cmap="jet")
        plt.axis("off"); plt.title("Grad-CAM")
    out = "reports/figures/gradcam_grid.png"
    plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="models/best_model.keras")
    ap.add_argument("--num", type=int, default=5)
    args = ap.parse_args()
    main(args.model_path, args.num)
