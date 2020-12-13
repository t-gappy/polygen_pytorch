import torch
from .base import Tokenizer


class FaceTokenizer(Tokenizer):
    
    def __init__(self, eof_id=0, eos_id=1, pad_id=2):
        self.special_tokens = {
            "eof": torch.tensor([eof_id]),
            "eos": torch.tensor([eos_id]),
            "pad": torch.tensor([pad_id]),
        }
        self.pad_id = pad_id
        self.not_coord_token = torch.tensor([0])
        
    def tokenize(self, faces, target=False, padding=True):
        special_tokens = self.special_tokens
        not_coord_token = self.not_coord_token
        
        
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

            position_ids = [
                torch.cat([
                    torch.arange(len(f)-1) // 4 + 1,
                    not_coord_token
                ])
                for f in faces
            ]
            
            faces = self._padding(faces, special_tokens["pad"])
            
            
            if target:
                faces = [torch.cat([f, special_tokens["pad"]])[1:] for f in faces]
                outputs = {
                    "value_tokens": torch.stack(faces)
                }
            else: 
                faces = torch.stack(faces)
                coord_type_tokens = torch.stack(self._padding(coord_type_tokens, not_coord_token))
                position_ids = torch.stack(self._padding(position_ids, not_coord_token))
                
                padding_mask = self._make_padding_mask(faces, self.pad_id)
                future_mask = self._make_future_mask(faces)
                
                cond_vertice = faces >= len(special_tokens)
                reference_vertices_mask = torch.where(cond_vertice, 1., 0.)
                reference_vertices_ids = torch.where(cond_vertice, faces-len(special_tokens), 0)
                reference_embed_mask = torch.where(cond_vertice, 0., 1.)
                reference_embed_ids = torch.where(cond_vertice, 0, faces)
                
                outputs = {
                    "value_tokens": faces,
                    "coord_type_tokens": coord_type_tokens,
                    "position_ids": position_ids,
                    "ref_v_mask": reference_vertices_mask,
                    "ref_v_ids": reference_vertices_ids,
                    "ref_e_mask": reference_embed_mask,
                    "ref_e_ids": reference_embed_ids,
                    "padding_mask": padding_mask,
                    "future_mask": future_mask,
                }
            
        else:
            faces_ids = []
            coord_type_tokens = []
            position_ids = []
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
                pos_ids = torch.cat([
                    torch.arange(len(f)-1) // 4 + 1,
                    not_coord_token
                ])
                
                if target:
                    f = torch.cat([f, special_tokens["pad"]])[1:]
                
                cond_vertice = f >= len(special_tokens)

                ref_v_mask = torch.where(cond_vertice, 1., 0.)
                ref_e_mask = torch.where(cond_vertice, 0., 1.)
                ref_v_ids = torch.where(cond_vertice, f-len(special_tokens), 0)
                ref_e_ids = torch.where(cond_vertice, 0, f)

                faces_ids.append(f)
                coord_type_tokens.append(c_t_tokens)
                position_ids.append(pos_ids)
                
                reference_vertices_mask.append(ref_v_mask)
                reference_vertices_ids.append(ref_v_ids)
                reference_embed_mask.append(ref_e_mask)
                reference_embed_ids.append(ref_e_ids)
            
            if target:
                faces_ids = [torch.cat([f, special_tokens["pad"]])[1:] for f in faces_ids]
                outputs = {
                    "value_tokens": faces_ids
                }
            else:
                outputs = {
                    "value_tokens": faces_ids,
                    "coord_type_tokens": coord_type_tokens,
                    "position_ids": position_ids,
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