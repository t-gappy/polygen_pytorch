import torch
import numpy as np


class Tokenizer(object):
    
    def _padding(self, ids_tensor, pad_token, max_length=None):
        if max_length is None:
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
            torch.zeros_like(ids_tensor),
            torch.ones_like(ids_tensor)
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
    
    def get_pred_start(self, batch_size=1):
        special_tokens = self.special_tokens
        not_coord_token = self.not_coord_token
        max_seq_len = self.max_seq_len
        
        vertices = torch.stack(
            self._padding(
                [special_tokens["bos"]] * batch_size, 
                special_tokens["pad"],
                max_seq_len
            )
        )
        coord_type_tokens = torch.stack(
            self._padding(
                [self.not_coord_token] * batch_size,
                not_coord_token,
                max_seq_len
            )
        )
        position_tokens = torch.stack(
            self._padding(
                [self.not_coord_token] * batch_size,
                not_coord_token,
                max_seq_len
            )
        )
        
        padding_mask = self._make_padding_mask(vertices, self.pad_id)
        
        outputs = {
            "value_tokens": vertices,
            "coord_type_tokens": coord_type_tokens,
            "position_tokens": position_tokens,
            "padding_mask": padding_mask,
        }
        return outputs
