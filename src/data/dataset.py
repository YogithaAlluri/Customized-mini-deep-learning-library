class Dataset:
    """
    Base class for datasets.
    """

    def __len__(self):
        raise NotImplementedError("Dataset must implement __len__")

    def __getitem__(self, idx):
        raise NotImplementedError("Dataset must implement __getitem__")

