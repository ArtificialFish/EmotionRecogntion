import torch

from dataset import get_train_test_loaders
from model import Model
from utils import *

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def gen_labels(loader, model):
    """Return true and predicted values."""
    y_true, y_pred = [], []
    for X, y in loader:
        with torch.no_grad():
            output = model(X)
            predicted = predictions(output.data)
            y_true = np.append(y_true, y.numpy())
            y_pred = np.append(y_pred, predicted.numpy())
    return y_true, y_pred


def plot_conf(loader, model, sem_labels, png_name):
    """Draw confusion matrix."""
    y_true, y_pred = gen_labels(loader, model)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues, interpolation="nearest")
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.set_label("Frequency", rotation=270, labelpad=10)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, z, ha="center", va="center")
    plt.gcf().text(0.02, 0.4, sem_labels, fontsize=9)
    plt.subplots_adjust(left=0.5)
    ax.set_xlabel("Predictions")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("True Labels")
    plt.savefig(png_name)


def main():
    """Create confusion matrix and save to file."""
    _, te_loader, _ = get_train_test_loaders(task="target", batch_size=32)

    model = Model()
    print("Loading target...")
    model, _, _ = restore_checkpoint(
        model, "./checkpoints/target/", force=True
    )

    sem_labels = "0 - Angry\n1 - Disgust\n2 - Fear\n3 - Happy\n4 - Neutral\n5 - Sad\n6 - Surprise"

    # Evaluate model
    plot_conf(te_loader, model, sem_labels, "conf_matrix.png")


if __name__ == "__main__":
    main()
