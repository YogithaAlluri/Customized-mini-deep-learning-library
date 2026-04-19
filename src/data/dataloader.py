import numpy as np


class DataLoader:
    """
    Simple DataLoader that returns batches from a Dataset.
    """

    def __init__(self, dataset, batch_size=1, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # Create an index list
        indices = np.arange(len(self.dataset))

        # Shuffle if needed
        if self.shuffle:
            np.random.shuffle(indices)

        # Yield batches
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]

            batch = [self.dataset[i] for i in batch_idx]

            # Unzip batch into x and y
            xs, ys = zip(*batch)

            yield xs, ys

