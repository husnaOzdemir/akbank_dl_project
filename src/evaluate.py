import os, json, numpy as np, matplotlib.pyplot as plt, seaborn as sns, tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from utils import CLASS_NAMES
from data import load_data, make_ds

def main(model_path="models/best_model.keras"):
    os.makedirs("reports/figures", exist_ok=True)
    (_,_,_), (_, yv, yv_oh), (xt, yt, yt_oh) = load_data()
    model = tf.keras.models.load_model(model_path)

    test_ds = make_ds(xt, yt_oh, training=False, batch_size=128)
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test Acc: {acc:.4f} | Loss: {loss:.4f}")

    y_prob = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(yt_oh, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    # CM görsel
    plt.figure(figsize=(14,12))
    sns.heatmap(cm_norm, cmap="Blues", cbar=True,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, square=True)
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    plt.title("Confusion Matrix (Normalized)")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png", dpi=140); plt.close()

    # Rapor
    rep = classification_report(y_true, y_pred, digits=4,
                                target_names=[f"{i}: {n}" for i,n in enumerate(CLASS_NAMES)])
    with open("reports/figures/classification_report.txt","w") as f: f.write(rep)
    print(rep)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="models/best_model.keras")
    args = ap.parse_args()
    main(args.model_path)
