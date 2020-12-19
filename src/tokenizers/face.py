import torch
from .base import Tokenizer


class FaceTokenizer(Tokenizer):
    
    def __init__(self, bof_id=0, eos_id=1, pad_id=2, max_seq_len=None):
        self.special_tokens = {
            "bof": torch.tensor([bof_id]),
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
        
        faces_ids = []
        in_position_tokens = []
        out_position_tokens = []
        faces_target = []

        for face in faces:
            face_with_bof = [
                torch.cat([
                    special_tokens["bof"],
                    f + len(special_tokens)
                ])
                for f in face
            ]
            face = torch.cat([
                torch.cat(face_with_bof),
                special_tokens["eos"]
            ])
            faces_ids.append(face)
            faces_target.append(torch.cat([face, special_tokens["pad"]])[1:])

            in_position_token = torch.cat([
                torch.arange(1, len(f)+1)
                for f in face_with_bof
            ])
            in_position_token = torch.cat([in_position_token, not_coord_token])
            in_position_tokens.append(in_position_token)

            out_position_token = torch.cat([
                torch.ones((len(f), ), dtype=torch.int32) * (idx+1)
                for idx, f in enumerate(face_with_bof)
            ])
            out_position_token = torch.cat([out_position_token, not_coord_token])
            out_position_tokens.append(out_position_token)
        
        
        if padding:
            faces_ids = torch.stack(
                self._padding(faces_ids, special_tokens["pad"], max_seq_len)
            )
            faces_target = torch.stack(
                self._padding(faces_target, special_tokens["pad"], max_seq_len)
            )
            in_position_tokens = torch.stack(
                self._padding(in_position_tokens, not_coord_token, max_seq_len)
            )
            out_position_tokens = torch.stack(
                self._padding(out_position_tokens, not_coord_token, max_seq_len)
            )

            padding_mask = self._make_padding_mask(faces_ids, self.pad_id)
            # future_mask = self._make_future_mask(faces)

            cond_vertice = faces_ids >= len(special_tokens)
            reference_vertices_mask = torch.where(cond_vertice, 1., 0.)
            reference_vertices_ids = torch.where(cond_vertice, faces_ids-len(special_tokens), 0)
            reference_embed_mask = torch.where(cond_vertice, 0., 1.)
            reference_embed_ids = torch.where(cond_vertice, 0, faces_ids)

            outputs = {
                "value_tokens": faces_ids,
                "target_tokens": faces_target,
                "in_position_tokens": in_position_tokens,
                "out_position_tokens": out_position_tokens,
                "ref_v_mask": reference_vertices_mask,
                "ref_v_ids": reference_vertices_ids,
                "ref_e_mask": reference_embed_mask,
                "ref_e_ids": reference_embed_ids,
                "padding_mask": padding_mask,
                # "future_mask": future_mask,
            }
            
        else:
            reference_vertices_mask = []
            reference_vertices_ids = []
            reference_embed_mask = []
            reference_embed_ids = []

            for f in faces_ids:
                cond_vertice = f >= len(special_tokens)

                ref_v_mask = torch.where(cond_vertice, 1., 0.)
                ref_e_mask = torch.where(cond_vertice, 0., 1.)
                ref_v_ids = torch.where(cond_vertice, f-len(special_tokens), 0)
                ref_e_ids = torch.where(cond_vertice, 0, f)
                
                reference_vertices_mask.append(ref_v_mask)
                reference_vertices_ids.append(ref_v_ids)
                reference_embed_mask.append(ref_e_mask)
                reference_embed_ids.append(ref_e_ids)
            
            outputs = {
                "value_tokens": faces_ids,
                "target_tokens": faces_target,
                "in_position_tokens": in_position_tokens,
                "out_position_tokens": out_position_tokens,
                "ref_v_mask": reference_vertices_mask,
                "ref_v_ids": reference_vertices_ids,
                "ref_e_mask": reference_embed_mask,
                "ref_e_ids": reference_embed_ids,
            }
        
        return outputs

    def tokenize_prediction(self, faces):
        special_tokens = self.special_tokens
        not_coord_token = self.not_coord_token
        max_seq_len = self.max_seq_len
        
        faces_ids = []
        in_position_tokens = []
        out_position_tokens = []
        faces_target = []    
        
        for face in faces:
            face = torch.cat([special_tokens["bof"], face])
            faces_ids.append(face)
            faces_target.append(torch.cat([face, special_tokens["pad"]])[1:])
            
            
            bof_indeces = torch.where(face==special_tokens["bof"])[0]
            now_pos_in = 1
            now_pos_out = 0
            in_position_token = []
            out_position_token = []
            
            for idx, point in enumerate(face):
                if idx in bof_indeces:
                    now_pos_out += 1
                    now_pos_in = 1
                
                in_position_token.append(now_pos_in)
                out_position_token.append(now_pos_out)
                now_pos_in += 1
                
            in_position_tokens.append(torch.tensor(in_position_token))
            out_position_tokens.append(torch.tensor(out_position_token))
            

        faces_ids = torch.stack(
            self._padding(faces_ids, special_tokens["pad"], max_seq_len)
        )
        faces_target = torch.stack(
            self._padding(faces_target, special_tokens["pad"], max_seq_len)
        )
        in_position_tokens = torch.stack(
            self._padding(in_position_tokens, not_coord_token, max_seq_len)
        )
        out_position_tokens = torch.stack(
            self._padding(out_position_tokens, not_coord_token, max_seq_len)
        )

        padding_mask = self._make_padding_mask(faces_ids, self.pad_id)
        # future_mask = self._make_future_mask(faces)

        cond_vertice = faces_ids >= len(special_tokens)
        reference_vertices_mask = torch.where(cond_vertice, 1., 0.)
        reference_vertices_ids = torch.where(cond_vertice, faces_ids-len(special_tokens), 0)
        reference_embed_mask = torch.where(cond_vertice, 0., 1.)
        reference_embed_ids = torch.where(cond_vertice, 0, faces_ids)

        outputs = {
            "value_tokens": faces_ids,
            "target_tokens": faces_target,
            "in_position_tokens": in_position_tokens,
            "out_position_tokens": out_position_tokens,
            "ref_v_mask": reference_vertices_mask,
            "ref_v_ids": reference_vertices_ids,
            "ref_e_mask": reference_embed_mask,
            "ref_e_ids": reference_embed_ids,
            "padding_mask": padding_mask,
            # "future_mask": future_mask,
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
    
    