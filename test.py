import torch

from dataset import get_train_test_loaders
from model import Model

from utils import *


def main():
    """Print performance metrics for model at specified epoch."""
    # Data loaders
    tr_loader, te_loader, _ = get_train_test_loaders(
        task="target",
        batch_size=32,
    )

    # Model
    model = Model()

    # Loss Function
    criterion = torch.nn.CrossEntropyLoss()

    # Attempts to restore the latest checkpoint if exists
    print("Loading cnn...")
    model, start_epoch, stats = restore_checkpoint(
        model, "./checkpoints/target/", force=True
    )

    axes = make_training_plot()

    # Evaluate the model
    eval_epoch(
        axes,
        tr_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        include_test=True,
        update_plot=False,
    )


if __name__ == "__main__":
    main()
