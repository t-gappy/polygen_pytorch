import torch
from .base import Tokenizer


class FaceTokenizer(Tokenizer):
    
    def __init__(self, eof_id=0, eos_id=1, pad_id=2, max_seq_len=None):
        self.special_tokens = {
            "eof": torch.tensor([eof_id]),
            "eos": torch.tensor([eos_id]),
            "pad": torch.tensor([pad_id]),
        }
        self.pad_id = pad_id
        self.not_coord_token = torch.tensor([0])
        if max_seq_len is not None:
            self.max_seq_len = max_seq_len - 1
        else:
            self.max_seq_len = max_seq_len
        
    def tokenize(self, faces, padding=True):
        special_tokens = self.special_tokens
        not_coord_token = self.not_coord_token
        max_seq_len = self.max_seq_len
        
        
        if padding:
            faces = [
                torch.cat([
                    torch.cat([
                        f + len(special_tokens),
                        special_tokens["eof"].repeat(len(f))[:, None]
                    ], dim=1).reshape(-1,),
                    special_tokens["eos"]
                ]) for f in faces
            ]
            
            coord_type_tokens = [
                torch.cat([
                    torch.arange(len(f)-1) % 4 + 1,
                    not_coord_token
                ])
                for f in faces
            ]

            position_tokens = [
                torch.cat([
                    torch.arange(len(f)-1) // 4 + 1,
                    not_coord_token
                ])
                for f in faces
            ]
            
            faces_target = [
                torch.cat([f, special_tokens["pad"]])[1:] 
                for f in faces
            ]
            
            faces = torch.stack(
                self._padding(faces, special_tokens["pad"], max_seq_len)
            )
            faces_target = torch.stack(
                self._padding(faces_target, special_tokens["pad"], max_seq_len)
            )
            coord_type_tokens = torch.stack(
                self._padding(coord_type_tokens, not_coord_token, max_seq_len)
            )
            position_tokens = torch.stack(
                self._padding(position_tokens, not_coord_token, max_seq_len)
            )

            padding_mask = self._make_padding_mask(faces, self.pad_id)
            # future_mask = self._make_future_mask(faces)

            cond_vertice = faces >= len(special_tokens)
            reference_vertices_mask = torch.where(cond_vertice, 1., 0.)
            reference_vertices_ids = torch.where(cond_vertice, faces-len(special_tokens), 0)
            reference_embed_mask = torch.where(cond_vertice, 0., 1.)
            reference_embed_ids = torch.where(cond_vertice, 0, faces)

            outputs = {
                "value_tokens": faces,
                "target_tokens": faces_target,
                "coord_type_tokens": coord_type_tokens,
                "position_tokens": position_tokens,
                "ref_v_mask": reference_vertices_mask,
                "ref_v_ids": reference_vertices_ids,
                "ref_e_mask": reference_embed_mask,
                "ref_e_ids": reference_embed_ids,
                "padding_mask": padding_mask,
                # "future_mask": future_mask,
            }
            
        else:
            faces_ids = []
            coord_type_tokens = []
            position_tokens = []
            reference_vertices_mask = []
            reference_vertices_ids = []
            reference_embed_mask = []
            reference_embed_ids = []

            for f in faces:
                f = torch.cat([
                    f + len(special_tokens),
                    special_tokens["eof"].repeat(len(f))[:, None]
                ], dim=1).reshape(-1, )
                f = torch.cat([f, special_tokens["eos"]])
                
                c_t_tokens = torch.cat([
                    torch.arange(len(f)-1) % 4 + 1,
                    not_coord_token
                ])
                pos_tokens = torch.cat([
                    torch.arange(len(f)-1) // 4 + 1,
                    not_coord_token
                ])
                
                cond_vertice = f >= len(special_tokens)

                ref_v_mask = torch.where(cond_vertice, 1., 0.)
                ref_e_mask = torch.where(cond_vertice, 0., 1.)
                ref_v_ids = torch.where(cond_vertice, f-len(special_tokens), 0)
                ref_e_ids = torch.where(cond_vertice, 0, f)

                faces_ids.append(f)
                coord_type_tokens.append(c_t_tokens)
                position_tokens.append(pos_tokens)
                
                reference_vertices_mask.append(ref_v_mask)
                reference_vertices_ids.append(ref_v_ids)
                reference_embed_mask.append(ref_e_mask)
                reference_embed_ids.append(ref_e_ids)
            
            faces_target = [torch.cat([f, special_tokens["pad"]])[1:] for f in faces_ids]
            outputs = {
                "value_tokens": faces_ids,
                "target_tokens": faces_target,
                "coord_type_tokens": coord_type_tokens,
                "position_tokens": position_tokens,
                "ref_v_mask": reference_vertices_mask,
                "ref_v_ids": reference_vertices_ids,
                "ref_e_mask": reference_embed_mask,
                "ref_e_ids": reference_embed_ids,
            }
        
        return outputs

    def detokenize(self, faces):
        special_tokens = self.special_tokens
        
        result = []
        for face in faces:
            face = face - len(special_tokens)
            result.append(
                face[torch.where(face >= 0)]
            )
        return result
    
    