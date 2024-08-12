import os
import torch
from torch.nn.functional import softmax
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics


def count_parameters(model):
    """Count number of learnable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def restore_checkpoint(
    model, checkpoint_dir, cuda=False, force=False, pretrain=False
):
    """Restore model from checkpoint if it exists.

    Returns the model and the current epoch.
    """
    try:
        cp_files = [
            file_
            for file_ in os.listdir(checkpoint_dir)
            if file_.startswith("epoch=")
            and file_.endswith(".checkpoint.pth.tar")
        ]
    except FileNotFoundError:
        cp_files = None
        os.makedirs(checkpoint_dir)
    if not cp_files:
        print("No saved model parameters found")
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0, []

    # Find latest epoch
    for i in itertools.count(1):
        if "epoch={}.checkpoint.pth.tar".format(i) in cp_files:
            epoch = i
        else:
            break

    if not force:
        print(
            f"Which epoch to load from? Choose in range [0, {epoch}].",
            "Enter 0 to train from scratch.",
        )
        print(">> ", end="")
        # inp_epoch = int(input())
        inp_epoch = 96
        if inp_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if inp_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0, []
    else:
        print(f"Which epoch to load from? Choose in range [1, {epoch}].")
        # inp_epoch = int(input())
        inp_epoch = 96
        if inp_epoch not in range(1, epoch + 1):
            raise Exception("Invalid epoch number")

    inp_path = f"epoch={inp_epoch}.checkpoint.pth.tar"
    filename = os.path.join(checkpoint_dir, inp_path)

    print("Loading from checkpoint {}?".format(filename))

    if cuda:
        checkpoint = torch.load(filename)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(
            filename, map_location=lambda storage, loc: storage
        )

    try:
        start_epoch = checkpoint["epoch"]
        stats = checkpoint["stats"]
        if pretrain:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["state_dict"])
        print(
             "=> Successfully restored checkpoint",
            f"(trained for {checkpoint["epoch"]} epochs)",
              )
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch, stats

def train_epoch(data_loader, model, criterion, optimizer):
    """Train the `model` for one epoch of data from `data_loader`.

    Use `optimizer` to optimize the specified `criterion`
    """
    for _, (X, y) in enumerate(data_loader):
        optimizer.zero_grad()

        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

def eval_epoch(
    axes,
    tr_loader,
    te_loader,
    model,
    criterion,
    epoch,
    stats,
    include_test=False,
    update_plot=True,
    multiclass=True,
):
    """Evaluate the `model` on the train and test set."""

    def _get_metrics(loader):
        y_true, y_pred, y_score = [], [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in loader:
            with torch.no_grad():
                output = model(X)
                predicted = predictions(output.data)
                y_true.append(y)
                y_pred.append(predicted)
                if not multiclass:
                    y_score.append(softmax(output.data, dim=1)[:, 1])
                else:
                    y_score.append(softmax(output.data, dim=1))
                total += y.size(0)
                correct += (predicted == y).sum().item()
                running_loss.append(criterion(output, y).item())
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        y_score = torch.cat(y_score)
        loss = np.mean(running_loss)
        acc = correct / total
        if not multiclass:
            auroc = metrics.roc_auc_score(y_true, y_score)
        else:
            auroc = metrics.roc_auc_score(y_true, y_score, multi_class="ovo")
        return acc, loss, auroc

    stats_at_epoch = list(_get_metrics(tr_loader))

    if include_test:
        stats_at_epoch += list(_get_metrics(te_loader))

    stats.append(stats_at_epoch)
    log_training(epoch, stats)
    if update_plot:
        update_training_plot(axes, epoch, stats)


def save_checkpoint(model, epoch, checkpoint_dir, stats):
    """Save a checkpoint file to `checkpoint_dir`."""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "stats": stats,
    }

    filename = os.path.join(
        checkpoint_dir, f"epoch={epoch}.checkpoint.pth.tar"
    )
    torch.save(state, filename)


def early_stopping(stats, curr_count_to_patience, global_min_loss):
    """Update count to patience and validation loss."""
    print(stats[len(stats) - 1][1])
    print(global_min_loss)
    if stats[len(stats) - 1][1] < global_min_loss:
        global_min_loss = stats[len(stats) - 1][1]
        curr_count_to_patience = 0
    else:
        curr_count_to_patience += 1

    return curr_count_to_patience, global_min_loss


def make_training_plot(name="CNN Training"):
    """Set up an interactive matplotlib graph to log metrics during training."""
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    plt.suptitle(name)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUROC")

    return axes


def predictions(logits):
    """Determine predicted class index given a tensor of logits.

    Example: Given tensor([[0.2, -0.8], [-0.9, -3.1], [0.5, 2.3]]), return tensor([0, 0, 1])

    Returns:
        the predicted class output as a PyTorch Tensor
    """
    pred = torch.argmax(logits, dim=1)

    return pred


def log_training(epoch, stats):
    """Print the train and test accuracy/loss/auroc."""
    splits = ["Train", "Test"]
    metrics = ["Accuracy", "Loss", "AUROC"]
    print("Epoch {}".format(epoch))
    for j, split in enumerate(splits):
        for i, metric in enumerate(metrics):
            idx = len(metrics) * j + i
            if idx >= len(stats[-1]):
                continue
            print(f"\t{split} {metric}:{round(stats[-1][idx],4)}")


def update_training_plot(axes, epoch, stats):
    """Update the training plot with a new data point for loss and accuracy."""
    splits = ["Train", "Test"]
    metrics = ["Accuracy", "Loss", "AUROC"]
    colors = ["r", "b", "g"]
    for i, _ in enumerate(metrics):
        for j, _ in enumerate(splits):
            idx = len(metrics) * j + i
            if idx >= len(stats[-1]):
                continue
            # __import__('pdb').set_trace()
            axes[i].plot(
                range(epoch - len(stats) + 1, epoch + 1),
                [stat[idx] for stat in stats],
                linestyle="--",
                marker="o",
                color=colors[j],
            )
        axes[i].legend(splits[: int(len(stats[-1]) / len(metrics))])
    plt.pause(0.00001)


def clear_checkpoint(checkpoint_dir):
    """Remove checkpoints in `checkpoint_dir`."""
    filelist = [
        f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")
    ]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")
