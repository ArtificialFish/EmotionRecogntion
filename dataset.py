import os
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pandas as pd
import torch

from cv2 import imread
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def get_train_test_loaders(task="target", batch_size=32, **kwargs):
    """Return DataLoaders for train, val and test splits.

    Any keyword arguments are forwarded to the LandmarksDataset constructor.
    """
    tr, te, _ = get_train_test_datasets(task, **kwargs)

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)

    return tr_loader, te_loader, tr.get_semantic_label


def get_train_test_datasets(task="target", **kwargs):
    """Return EmotionDatasets and image standardizer.

    Image standardizer should be fit to train data and applied to all splits.
    """
    tr = EmotionDataset("train", task, **kwargs)
    te = EmotionDataset("test", task, **kwargs)

    # Resize
    tr.X = resize(tr.X)
    te.X = resize(te.X)

    # Standardize
    standardizer = ImageStandardizer()
    standardizer.fit(tr.X)
    tr.X = standardizer.transform(tr.X)
    te.X = standardizer.transform(te.X)

    # Transpose the dimensions from (N,H,W,C) to (N,C,H,W)
    tr.X = tr.X.transpose(0, 3, 1, 2)
    te.X = te.X.transpose(0, 3, 1, 2)

    return tr, te, standardizer


class EmotionDataset(Dataset):
    """Dataset class for emotion images."""

    def __init__(self, partition, task="target"):
        """Initialize dataset."""
        super().__init__()

        if partition not in ["train", "test"]:
            raise ValueError(f"Partition {partition} does not exist")

        self.partition = partition
        self.task = task

        self.csv = pd.read_csv(f"data/emotions.csv")
        self.X, self.y = self._load_data()

        self.semantic_labels = dict(
            zip(
                self.csv[self.csv.task == self.task]["numeric_label"],
                self.csv[self.csv.task == self.task]["semantic_label"],
            )
        )

    def __len__(self):
        """Return size of dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return (image, label) pair at index `idx` of dataset."""
        return (
            torch.from_numpy(self.X[idx]).float(),
            torch.tensor(self.y[idx]).long(),
        )

    def _load_data(self):
        """Load a single data partition from file."""
        print(f"loading {self.partition}...")
        df = self.csv[
            (self.csv.task == self.task)
            & (self.csv.partition == self.partition)
        ]
        path = f"data/images/{self.partition}/"

        X, y = [], []
        for _, row in df.iterrows():
            image = imread(os.path.join(path, row["filename"]))
            X.append(image)
            y.append(row["numeric_label"])
        return np.array(X), np.array(y)

    def get_semantic_label(self, numeric_label):
        """Return the string representation of the numeric class label."""
        return self.semantic_labels[numeric_label]


def resize(X):
    """Resize the data partition X to the size specified in the config file."""
    image_size = (64, 64)
    resized = []
    for i in range(X.shape[0]):
        xi = Image.fromarray(X[i].astype(np.uint8)).resize(
            image_size, resample=2
        )
        resized.append(xi)
    resized = [np.asarray(im) for im in resized]
    resized = np.array(resized)

    return resized


class ImageStandardizer(object):
    """Standardize a batch of images to mean 0 and variance 1."""

    def __init__(self):
        """Initialize mean and standard deviations to None."""
        super().__init__()
        self.image_mean = None
        self.image_std = None

    def fit(self, X):
        """Calculate per-channel mean and standard deviation from dataset X."""

        self.image_mean = X.mean(axis=(0, 1, 2))
        self.image_std = X.std(axis=(0, 1, 2))

        print(self.image_mean)
        print(self.image_std)

    def transform(self, X):
        """Return standardized dataset given dataset X."""
        return (X - self.image_mean) / self.image_std


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    tr, te, standardizer = get_train_test_datasets(task="target")
    print("Train:\t", len(tr.X))
    print("Test:\t", len(te.X))
    print("Mean:", standardizer.image_mean)
    print("Std: ", standardizer.image_std)
