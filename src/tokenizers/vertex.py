import torch
from .base import Tokenizer


class EncodeVertexTokenizer(Tokenizer):
    
    def __init__(self, pad_id=0):
        self.pad_token = torch.tensor([pad_id])
        self.pad_id = pad_id
        
    def tokenize(self, vertices, padding=True):
        vertices = [v.reshape(-1,) + 1 for v in vertices]
        coord_type_tokens = [torch.arange(len(v)) % 3 + 1 for v in vertices]
        position_tokens = [torch.arange(len(v)) // 3 + 1 for v in vertices]
        
        if padding:
            vertices = torch.stack(self._padding(vertices, self.pad_token))
            coord_type_tokens = torch.stack(self._padding(coord_type_tokens, self.pad_token))
            position_tokens = torch.stack(self._padding(position_tokens, self.pad_token))
            padding_mask = self._make_padding_mask(vertices, self.pad_id)
            
            outputs = {
                "value_tokens": vertices,
                "coord_type_tokens": coord_type_tokens,
                "position_tokens": position_tokens,
                "padding_mask": padding_mask,
            }
        else:
            outputs = {
                "value_tokens": vertices,
                "coord_type_tokens": coord_type_tokens,
                "position_tokens": position_tokens,
            }
            
        return outputs

    
    
class DecodeVertexTokenizer(Tokenizer):
    
    def __init__(self, bos_id=0, eos_id=1, pad_id=2, max_seq_len=None):
        
        self.special_tokens = {
            "bos": torch.tensor([bos_id]),
            "eos": torch.tensor([eos_id]),
            "pad": torch.tensor([pad_id]),
        }
        self.pad_id = pad_id
        self.not_coord_token = torch.tensor([0])
        if max_seq_len is not None:
            self.max_seq_len = max_seq_len - 1
        else:
            self.max_seq_len = max_seq_len
        
    
    def tokenize(self, vertices, padding=True):
        special_tokens = self.special_tokens
        not_coord_token = self.not_coord_token
        max_seq_len = self.max_seq_len
        
        vertices = [
            torch.cat([
                special_tokens["bos"], 
                v.reshape(-1,)  + len(special_tokens), 
                special_tokens["eos"]
            ])
            for v in vertices
        ]
        
        coord_type_tokens = [
            torch.cat([
                not_coord_token,
                torch.arange(len(v)-2) % 3 + 1,
                not_coord_token
            ])
            for v in vertices
        ]
        
        position_tokens = [
            torch.cat([
                not_coord_token,
                torch.arange(len(v)-2) // 3 + 1,
                not_coord_token
            ])
            for v in vertices
        ]
        
        vertices_target = [
            torch.cat([v, special_tokens["pad"]])[1:] 
            for v in vertices
        ]
        
        if padding:
            vertices = torch.stack(
                self._padding(vertices, special_tokens["pad"], max_seq_len)
            )
            vertices_target = torch.stack(
                self._padding(vertices_target, special_tokens["pad"], max_seq_len)
            )
            coord_type_tokens = torch.stack(
                self._padding(coord_type_tokens, not_coord_token, max_seq_len)
            )
            position_tokens = torch.stack(
                self._padding(position_tokens, not_coord_token, max_seq_len)
            )
            
            padding_mask = self._make_padding_mask(vertices, self.pad_id)
            # future_mask = self._make_future_mask(vertices)
            outputs = {
                "value_tokens": vertices,
                "target_tokens": vertices_target,
                "coord_type_tokens": coord_type_tokens,
                "position_tokens": position_tokens,
                "padding_mask": padding_mask,
                # "future_mask": future_mask,
            }
        else:
            outputs = {
                "value_tokens": vertices,
                "target_tokens": vertices_target,
                "coord_type_tokens": coord_type_tokens,
                "position_tokens": position_tokens,
            }
            
        return outputs
    
    def detokenize(self, vertices):
        special_tokens = self.special_tokens
        
        result = []
        for vertex in vertices:
            vertex = vertex - len(special_tokens)
            result.append(
                vertex[torch.where(vertex >= 0)]
            )
        return result
    
