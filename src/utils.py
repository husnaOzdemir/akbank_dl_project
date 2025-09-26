
import os, json, random, numpy as np, matplotlib.pyplot as plt

SEED = 42

CLASS_NAMES = [
    "Speed limit (20km/h)","Speed limit (30km/h)","Speed limit (50km/h)","Speed limit (60km/h)",
    "Speed limit (70km/h)","Speed limit (80km/h)","End of speed limit (80km/h)","Speed limit (100km/h)",
    "Speed limit (120km/h)","No passing","No passing for >3.5t","Right-of-way at next intersection",
    "Priority road","Yield","Stop","No vehicles",">3.5t prohibited","No entry","General caution",
    "Dangerous curve left","Dangerous curve right","Double curve","Bumpy road","Slippery road",
    "Road narrows right","Road work","Traffic signals","Pedestrians","Children crossing","Bicycles crossing",
    "Beware of ice/snow","Wild animals crossing","End of all limits","Turn right ahead","Turn left ahead",
    "Ahead only","Go straight or right","Go straight or left","Keep right","Keep left","Roundabout mandatory",
    "End of no passing","End of no passing >3.5t"
]

def set_seeds(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

def plot_training(history, out_path=None):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history["accuracy"], label="train_acc")
    plt.plot(history["val_accuracy"], label="val_acc"); plt.legend(); plt.title("Accuracy")
    plt.subplot(1,2,2)
    plt.plot(history["loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss"); plt.legend(); plt.title("Loss")
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()

def save_history(hist_obj, path="history.json"):
    with open(path, "w") as f:
        json.dump({k: list(map(float, v)) for k, v in hist_obj.history.items()}, f)
