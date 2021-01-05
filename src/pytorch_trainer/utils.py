import torch

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        if len(x) != len(y):
            msg = "len(x) and len(y) must be the same"
            raise ValueError(msg)

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        return x, y


def collate_fn(batch):
    tweets = [xy[0] for xy in  batch]
    targets = [xy[1] for xy in batch]
    return tweets, targets
