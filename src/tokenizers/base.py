import torch
import numpy as np


class Tokenizer(object):
    
    def _padding(self, ids_tensor, pad_token):
        max_length = max([len(ids) for ids in ids_tensor])
        ids_tensor = [
            torch.cat([
                ids, pad_token.repeat(max_length - len(ids) + 1)
            ])
            for ids in ids_tensor
        ]
        return ids_tensor
    
    def _make_padding_mask(self, ids_tensor, pad_id):
        mask = torch.where(
            ids_tensor==pad_id,
            torch.ones_like(ids_tensor),
            torch.zeros_like(ids_tensor)
        ).type(torch.bool)
        return mask

    def _make_future_mask(self, ids_tensor):
        batch, length = ids_tensor.shape
        arange = torch.arange(length)
        mask = torch.where(
            arange[None, :] <= arange[:, None],
            torch.zeros((length, length)),
            torch.ones((length, length))*(-np.inf)
        ).type(torch.float32)
        return mask