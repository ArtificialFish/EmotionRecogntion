import torch

import matplotlib.pyplot as plt

from dataset import get_train_test_loaders
from model import Model
from utils import *


def main():
    """Train CNN and show training plots."""
    # Data loaders
    tr_loader, te_loader, _ = get_train_test_loaders(
        task="target",
        batch_size=64,
    )

    # Model
    model = Model()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 6.5e-4)

    print("Number of float-valued parameters:", count_parameters(model))

    # Attempts to restore the latest checkpoint if exists
    print("Loading cnn...")
    model, start_epoch, stats = restore_checkpoint(
        model, "./checkpoints/target/"
    )

    axes = make_training_plot()

    # Evaluate the randomly initialized model
    eval_epoch(
        axes,
        tr_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
    )

    # initial val loss for early stopping
    global_min_loss = stats[0][1]

    patience = 8
    curr_count_to_patience = 0

    # Loop over the entire dataset multiple times
    epoch = start_epoch

    while curr_count_to_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        eval_epoch(
            axes,
            tr_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
            include_test=False,
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, "./checkpoints/target/", stats)

        # update early stopping parameters
        curr_count_to_patience, global_min_loss = early_stopping(
            stats, curr_count_to_patience, global_min_loss
        )

        epoch += 1
    print("Finished Training")
    # Save figure and keep plot open
    plt.savefig("cnn_training_plot.png", dpi=200)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
