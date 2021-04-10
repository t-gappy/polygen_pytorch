import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from reformer_pytorch import Reformer

from .utils import Config, accuracy
sys.path.append(os.path.dirname(os.getcwd()))
from tokenizers import EncodeVertexTokenizer, FaceTokenizer


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
    if type(m) == nn.Embedding:
        nn.init.uniform_(m.weight, -0.05, 0.05)
        


class FacePolyGenConfig(Config):
    
    def __init__(self,
                 embed_dim=256, 
                 src__max_seq_len=2400, 
                 src__tokenizer__pad_id=0,
                 tgt__max_seq_len=5600,
                 tgt__tokenizer__bof_id=0,
                 tgt__tokenizer__eos_id=1, 
                 tgt__tokenizer__pad_id=2,
                 src__embedding__vocab_value=256+3, 
                 src__embedding__vocab_coord_type=4, 
                 src__embedding__vocab_position=1000, 
                 src__embedding__pad_idx_value=2,
                 src__embedding__pad_idx_coord_type=0,
                 src__embedding__pad_idx_position=0,
                 tgt__embedding__vocab_value=3,
                 tgt__embedding__vocab_in_position=350,
                 tgt__embedding__vocab_out_position=2000,
                 tgt__embedding__pad_idx_value=2,
                 tgt__embedding__pad_idx_in_position=0,
                 tgt__embedding__pad_idx_out_position=0,
                 src__reformer__depth=12,
                 src__reformer__heads=8,
                 src__reformer__n_hashes=8,
                 src__reformer__bucket_size=48,
                 src__reformer__causal=True,
                 src__reformer__lsh_dropout=0.2, 
                 src__reformer__ff_dropout=0.2,
                 src__reformer__post_attn_dropout=0.2,
                 src__reformer__ff_mult=4,
                 tgt__reformer__depth=12,
                 tgt__reformer__heads=8,
                 tgt__reformer__n_hashes=8,
                 tgt__reformer__bucket_size=48,
                 tgt__reformer__causal=True,
                 tgt__reformer__lsh_dropout=0.2, 
                 tgt__reformer__ff_dropout=0.2,
                 tgt__reformer__post_attn_dropout=0.2,
                 tgt__reformer__ff_mult=4):
        
        # auto padding for max_seq_len
        src_denominator = (src__reformer__bucket_size * 2 * 3)
        if src__max_seq_len % src_denominator != 0:
            divisables = src__max_seq_len // src_denominator + 1
            src__max_seq_len_new = divisables * src_denominator
            print("src__max_seq_len changed, because of lsh-attention's bucket_size")
            print("before: {} --> after: {} (with bucket_size: {})".format(
                src__max_seq_len, src__max_seq_len_new, src__reformer__bucket_size
            ))
            src__max_seq_len = src__max_seq_len_new
            
        tgt_denominator = tgt__reformer__bucket_size * 2
        if tgt__max_seq_len % tgt_denominator != 0:
            divisables = tgt__max_seq_len // tgt_denominator + 1
            tgt__max_seq_len_new = divisables * tgt_denominator
            print("tgt__max_seq_len changed, because of lsh-attention's bucket_size")
            print("before: {} --> after: {} (with bucket_size: {})".format(
                tgt__max_seq_len, tgt__max_seq_len_new, tgt__reformer__bucket_size
            ))
            tgt__max_seq_len = tgt__max_seq_len_new
        
        
        # tokenizer config
        src_tokenizer_config = {
            "pad_id": src__tokenizer__pad_id,
            "max_seq_len": src__max_seq_len,
        }
        tgt_tokenizer_config = {
            "bof_id": tgt__tokenizer__bof_id,
            "eos_id": tgt__tokenizer__eos_id,
            "pad_id": tgt__tokenizer__pad_id,
            "max_seq_len": tgt__max_seq_len,
        }
        
        # embedding config
        src_embedding_config = {
            "vocab_value": src__embedding__vocab_value,
            "vocab_coord_type": src__embedding__vocab_coord_type,
            "vocab_position": src__embedding__vocab_position,
            "pad_idx_value": src__embedding__pad_idx_value,
            "pad_idx_coord_type": src__embedding__pad_idx_coord_type,
            "pad_idx_position": src__embedding__pad_idx_position,
            "embed_dim": embed_dim,
        }
        tgt_embedding_config = {
            "vocab_value": tgt__embedding__vocab_value,
            "vocab_in_position": tgt__embedding__vocab_in_position,
            "vocab_out_position": tgt__embedding__vocab_out_position,
            "pad_idx_value": tgt__embedding__pad_idx_value,
            "pad_idx_in_position": tgt__embedding__pad_idx_in_position,
            "pad_idx_out_position": tgt__embedding__pad_idx_out_position,
            "embed_dim": embed_dim,
        }
        
        # reformer info
        src_reformer_config = {
            "dim": embed_dim,
            "max_seq_len": src__max_seq_len,
            "depth": src__reformer__depth,
            "heads": src__reformer__heads,
            "bucket_size": src__reformer__bucket_size,
            "n_hashes": src__reformer__n_hashes,
            "causal": src__reformer__causal,
            "lsh_dropout": src__reformer__lsh_dropout, 
            "ff_dropout": src__reformer__ff_dropout,
            "post_attn_dropout": src__reformer__post_attn_dropout,
            "ff_mult": src__reformer__ff_mult,
        }
        
        tgt_reformer_config = {
            "dim": embed_dim,
            "max_seq_len": tgt__max_seq_len,
            "depth": tgt__reformer__depth,
            "heads": tgt__reformer__heads,
            "bucket_size": tgt__reformer__bucket_size,
            "n_hashes": tgt__reformer__n_hashes,
            "causal": tgt__reformer__causal,
            "lsh_dropout": tgt__reformer__lsh_dropout, 
            "ff_dropout": tgt__reformer__ff_dropout,
            "post_attn_dropout": tgt__reformer__post_attn_dropout,
            "ff_mult": tgt__reformer__ff_mult,
        }
        
        self.config = {
            "embed_dim": embed_dim,
            "src_tokenizer": src_tokenizer_config,
            "tgt_tokenizer": tgt_tokenizer_config,
            "src_embedding": src_embedding_config,
            "tgt_embedding": tgt_embedding_config,
            "src_reformer": src_reformer_config,
            "tgt_reformer": tgt_reformer_config,
        }

        
        

class FaceEncoderEmbedding(nn.Module):
    
    def __init__(self, embed_dim=256,
                 vocab_value=259, pad_idx_value=2, 
                 vocab_coord_type=4, pad_idx_coord_type=0,
                 vocab_position=1000, pad_idx_position=0):
        
        super().__init__()
        
        self.value_embed = nn.Embedding(
            vocab_value, embed_dim, padding_idx=pad_idx_value
        )
        self.coord_type_embed = nn.Embedding(
            vocab_coord_type, embed_dim, padding_idx=pad_idx_coord_type
        )
        self.position_embed = nn.Embedding(
            vocab_position, embed_dim, padding_idx=pad_idx_position
        )
        
        self.embed_scaler = math.sqrt(embed_dim)
        
    def forward(self, tokens):
        
        """get embedding for Face Encoder.
        
        Args
            tokens [dict]: tokenized vertex info.
                `value_tokens` [torch.tensor]:
                        padded (batch, length) shape long tensor
                        with coord value from 0 to 2^n(bit).
                `coord_type_tokens` [torch.tensor]:
                        padded (batch, length) shape long tensor implies x or y or z.
                `position_tokens` [torch.tensor]:
                        padded (batch, length) shape long tensor
                        representing coord position (NOT sequence position).
        
        Returns
            embed [torch.tensor]: (batch, length, embed) shape tensor after embedding.
                        
        """
        
        embed = self.value_embed(tokens["value_tokens"]) * self.embed_scaler
        embed = embed + (self.coord_type_embed(tokens["coord_type_tokens"]) * self.embed_scaler)
        embed = embed + (self.position_embed(tokens["position_tokens"]) * self.embed_scaler)
        
        embed = embed[:, :-1]
        embed = torch.cat([
            e.sum(dim=1).unsqueeze(dim=1) for e in embed.split(3, dim=1)
        ], dim=1)
        
        return embed
    
    def forward_original(self, tokens):
        # original PolyGen embedding did something like this (no position info?).
        embed = self.value_embed(tokens["value_tokens"]) * self.embed_scaler
        embed = torch.cat([
            e.sum(dim=1).unsqueeze(dim=1) for e in embed[:, :-1].split(3, dim=1)
        ], dim=1)
        return embed
    
    

class FaceDecoderEmbedding(nn.Module):
    
    def __init__(self, embed_dim=256,
                 vocab_value=3, pad_idx_value=2, 
                 vocab_in_position=100, pad_idx_in_position=0,
                 vocab_out_position=1000, pad_idx_out_position=0):
        
        super().__init__()
        
        self.value_embed = nn.Embedding(
            vocab_value, embed_dim, padding_idx=pad_idx_value
        )
        self.in_position_embed = nn.Embedding(
            vocab_in_position, embed_dim, padding_idx=pad_idx_in_position
        )
        self.out_position_embed = nn.Embedding(
            vocab_out_position, embed_dim, padding_idx=pad_idx_out_position
        )
        
        self.embed_scaler = math.sqrt(embed_dim)
        
    def forward(self, encoder_embed, tokens):
        
        """get embedding for Face Decoder.
        note that value_embeddings consist of two embedding.
          - pointer to encoder outputs
          - embedding for special tokens such as <end-of-face>, <eos>, <pad>.
        
        Args
            encoder_embed [torch.tensor]:
                    (batch, src-length, embed) shape tensor from encoder.
            tokens [dict]: all contents are in the shape of (batch, tgt-length).
                `ref_v_ids` [torch.tensor]:
                        this is used as pointer to `encoder_embed`.
                `ref_v_mask` [torch.tensor]:
                        mask for special token positions in pointer embeddings. 
                `ref_e_ids` [torch.tensor]:
                        embed ids for special tokens.
                `ref_e_ids` [torch.tensor]:
                        mask for pointer token position in special token embeddings.
                `in_position_tokens` [torch.tensor]:
                        embed ids for positions in face.
                `out_position_tokens` [torch.tensor]:
                        embed ids for positions of face itself in sequence.
                        
        Returns
            embed [torch.tensor]: (batch, tgt-length, embed) shape tensor of embeddings.
                        
        """
        
        embed = torch.cat([
            encoder_embed[b_idx, ids].unsqueeze(dim=0) 
            for b_idx, ids in enumerate(tokens["ref_v_ids"].unbind(dim=0))
        ], dim=0)
        embed = embed * tokens["ref_v_mask"].unsqueeze(dim=2)
        
        embed = (embed + \
                (self.value_embed(tokens["ref_e_ids"]) * 
                 self.embed_scaler *
                 tokens["ref_e_mask"].unsqueeze(dim=2)))
        
        embed = embed + (self.in_position_embed(tokens["in_position_tokens"]) * self.embed_scaler)
        embed = embed + (self.out_position_embed(tokens["out_position_tokens"]) * self.embed_scaler)
        return embed
    
    


class FacePolyGen(nn.Module):
    
    def __init__(self, model_config):
        super().__init__()
        self.src_tokenizer = EncodeVertexTokenizer(**model_config["src_tokenizer"])
        self.tgt_tokenizer = FaceTokenizer(**model_config["tgt_tokenizer"])
        
        self.src_embedding = FaceEncoderEmbedding(**model_config["src_embedding"])
        self.tgt_embedding = FaceDecoderEmbedding(**model_config["tgt_embedding"])
        
        self.src_reformer = Reformer(**model_config["src_reformer"])
        self.tgt_reformer = Reformer(**model_config["tgt_reformer"])
        
        self.src_norm = nn.LayerNorm(model_config["embed_dim"])
        self.tgt_norm = nn.LayerNorm(model_config["embed_dim"])
        self.loss_func = nn.CrossEntropyLoss(ignore_index=model_config["tgt_tokenizer"]["pad_id"])
        
        self.apply(init_weights)
        self.embed_scaler = math.sqrt(model_config["embed_dim"])
    
    def forward(self, src_tokens, device=None):
        
        """forward function which can be used for both train/predict.
        this function only encodes vertex information
        because decoders behave as really auto-regressive function.
        
        Args
            src_tokens [dict]: tokenized vertex info.
                `value_tokens` [torch.tensor]:
                        padded (batch, src-length) shape long tensor
                        with coord value from 0 to 2^n(bit).
                `coord_type_tokens` [torch.tensor]:
                        padded (batch, src-length) shape long tensor implies x or y or z.
                `position_tokens` [torch.tensor]:
                        padded (batch, src-length) shape long tensor
                        representing coord position (NOT sequence position).
                `padding_mask` [torch.tensor]:
                        (batch, src-length) shape mask implies <pad> tokens.
        
        Returns
            hs [torch.tensor]: (batch, src-length, embed) shape tensor after encoder.
        
        """
        
        hs = self.src_embedding(src_tokens)
        hs = self.src_reformer(
            hs, input_mask=src_tokens["padding_mask"]
        )
        hs = self.src_norm(hs)
        
        return hs
        
    def __call__(self, inputs, device=None):
        
        """Calculate loss while training.
        
        Args
            inputs [dict]: dict containing batched inputs.
                `vertices` [list(torch.tensor)]:
                        variable-length-list of 
                        (length, 3) shaped tensor of quantized-vertices.
                `faces` [list(list(torch.tensor))]:
                        batch-length-list of
                        variable-length-list (per face) of 
                        (length,) shaped vertex-ids which constructs a face.
            device [torch.device]: gpu or not gpu, that's the problem.
                
        Returns
            outputs [dict]: dict containing calculated variables.
                `loss` [torch.tensor]:
                        calculated scalar-shape loss with backprop info.
                `accuracy` [torch.tensor]:
                        calculated scalar-shape accuracy.
            
        """
        
        src_tokens = self.src_tokenizer.tokenize(inputs["vertices"])
        src_tokens = {k: v.to(device) for k, v in src_tokens.items()}
        
        tgt_tokens = self.tgt_tokenizer.tokenize(inputs["faces"])
        tgt_tokens = {k: v.to(device) for k, v in tgt_tokens.items()}
        
        encoder_embed = self.forward(src_tokens, device=device)
        hs = self.tgt_embedding(encoder_embed, tgt_tokens)
        hs = self.tgt_reformer(
            hs, input_mask=tgt_tokens["padding_mask"]
        )
        hs = self.tgt_norm(hs)
        
        # calc pointing to vertex
        BATCH = hs.shape[0]
        sptk_embed = self.tgt_embedding.value_embed.weight * self.embed_scaler
        encoder_embed = torch.cat([
            sptk_embed[None, ...].repeat(BATCH, 1, 1),
            encoder_embed
        ], dim=1)
        hs = torch.bmm(hs, encoder_embed.permute(0, 2, 1))
        
        BATCH, TGT_LENGTH, SRC_LENGTH = hs.shape
        hs = hs.reshape(BATCH*TGT_LENGTH, SRC_LENGTH)
        targets = tgt_tokens["target_tokens"].reshape(BATCH*TGT_LENGTH,)
        
        acc = accuracy(
            hs, targets, ignore_label=self.tgt_tokenizer.pad_id, device=device
        )
        loss = self.loss_func(hs, targets)
        
        if hasattr(self, 'reporter'):
            self.reporter.report({
                "accuracy": acc.item(),
                "perplexity": torch.exp(loss).item(),
                "loss": loss.item(),
            })

        return loss
    
    @torch.no_grad()
    def predict(self, inputs, max_seq_len=3936, device=None):
        tgt_tokenizer = self.tgt_tokenizer
        special_tokens = tgt_tokenizer.special_tokens
        
        # calc vertex encoding first.
        src_tokens = self.src_tokenizer.tokenize(inputs["vertices"])
        src_tokens = {k: v.to(device) for k, v in src_tokens.items()}
        
        encoder_embed = self.forward(src_tokens, device=device)
        BATCH = encoder_embed.shape[0]
        sptk_embed = self.tgt_embedding.value_embed.weight * self.embed_scaler
        encoder_embed_with_sptk = torch.cat([
            sptk_embed[None, ...].repeat(BATCH, 1, 1),
            encoder_embed
        ], dim=1)
        
        # prepare for generation.
        tgt_tokens = model.tgt_tokenizer.tokenize([[torch.tensor([], dtype=torch.int32)]])
        tgt_tokens["value_tokens"][:, 1] = model.tgt_tokenizer.special_tokens["pad"]
        tgt_tokens["ref_e_ids"][:, 1] = model.tgt_tokenizer.special_tokens["pad"]
        tgt_tokens["padding_mask"][:, 1] = True
        
        preds = [torch.tensor([], dtype=torch.int32)]
        pred_idx = 0
        
        now_face_idx = 0
        
        while (pred_idx <= max_seq_len-1) \
        and ((len(preds[now_face_idx]) == 0)
            or (preds[now_face_idx][-1] != special_tokens["eos"])):
            
            if pred_idx >= 1:
                tgt_tokens = tgt_tokenizer.tokenize([[torch.cat([p]) for p in preds]])
                tgt_tokens["value_tokens"][:, pred_idx+1] = special_tokens["pad"]
                tgt_tokens["ref_e_ids"][:, pred_idx+1] = special_tokens["pad"]
                tgt_tokens["padding_mask"][:, pred_idx+1] = True
            
            try:
                hs = self.tgt_embedding(encoder_embed, tgt_tokens)
            except IndexError:
                print("pred_ids", pred_idx)
                for k, v in tgt_tokens.items():
                    print(k)
                    print(v.shape)
                    print(v[:, pred_idx-5:pred_idx+2])
                print(preds)
                raise IndexError
                
            hs = self.tgt_reformer(
                hs, input_mask=tgt_tokens["padding_mask"]
            )
            hs = self.tgt_norm(hs)
            
            hs = torch.bmm(
                hs[:, pred_idx:pred_idx+1],
                encoder_embed_with_sptk.permute(0, 2, 1)
            )
            pred = hs[:, 0].argmax(dim=1)
            
            if pred[0] == special_tokens["bof"]:
                now_face_idx += 1
                preds.append(torch.tensor([], dtype=torch.int32))
            else:
                preds[now_face_idx] = \
                    torch.cat([preds[now_face_idx], pred[0, None]-len(special_tokens)])
            pred_idx += 1
            
        preds = torch.cat(preds) + len(special_tokens)
        preds = self.tgt_tokenizer.detokenize([preds])[0]    
        
        return preds
