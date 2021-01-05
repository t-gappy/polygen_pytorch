import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from reformer_pytorch import Reformer

from .utils import Config, accuracy
sys.path.append(os.path.dirname(os.getcwd()))
from tokenizers import DecodeVertexTokenizer


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
    if type(m) == nn.Embedding:
        nn.init.uniform_(m.weight, -0.05, 0.05)
        


class VertexPolyGenConfig(Config):
    
    def __init__(self,
                 embed_dim=256, 
                 max_seq_len=2400, 
                 tokenizer__bos_id=0,
                 tokenizer__eos_id=1,
                 tokenizer__pad_id=2,
                 embedding__vocab_value=256 + 3, 
                 embedding__vocab_coord_type=4, 
                 embedding__vocab_position=1000,
                 embedding__pad_idx_value=2,
                 embedding__pad_idx_coord_type=0,
                 embedding__pad_idx_position=0,
                 reformer__depth=12,
                 reformer__heads=8,
                 reformer__n_hashes=8,
                 reformer__bucket_size=48,
                 reformer__causal=True,
                 reformer__lsh_dropout=0.2, 
                 reformer__ff_dropout=0.2,
                 reformer__post_attn_dropout=0.2,
                 reformer__ff_mult=4):
        
        # tokenizer config
        tokenizer_config = {
            "bos_id": tokenizer__bos_id,
            "eos_id": tokenizer__eos_id,
            "pad_id": tokenizer__pad_id,
            "max_seq_len": max_seq_len,
        }
        
        # embedding config
        embedding_config = {
            "vocab_value": embedding__vocab_value,
            "vocab_coord_type": embedding__vocab_coord_type,
            "vocab_position": embedding__vocab_position,
            "pad_idx_value": embedding__pad_idx_value,
            "pad_idx_coord_type": embedding__pad_idx_coord_type,
            "pad_idx_position": embedding__pad_idx_position,
            "embed_dim": embed_dim,
        }
        
        # reformer info
        reformer_config = {
            "dim": embed_dim,
            "depth": reformer__depth,
            "max_seq_len": max_seq_len,
            "heads": reformer__heads,
            "bucket_size": reformer__bucket_size,
            "n_hashes": reformer__n_hashes,
            "causal": reformer__causal,
            "lsh_dropout": reformer__lsh_dropout, 
            "ff_dropout": reformer__ff_dropout,
            "post_attn_dropout": reformer__post_attn_dropout,
            "ff_mult": reformer__ff_mult,
        }
        
        self.config = {
            "embed_dim": embed_dim,
            "max_seq_len": max_seq_len,
            "tokenizer": tokenizer_config,
            "embedding": embedding_config,
            "reformer": reformer_config,
        }


class VertexDecoderEmbedding(nn.Module):
    
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
        
        """get embedding for vertex model.
        
        Args
            tokens [dict]: tokenized vertex info.
                `value_tokens` [torch.tensor]:
                        padded (batch, length)-shape long tensor
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
        
        return embed
    
    

class VertexPolyGen(nn.Module):
    
    """Vertex model in PolyGen.
    this model learn/predict vertices like OpenAI-GPT.
    UNLIKE the paper, this model is only for unconditional generation.
    
    Args
        model_config [Config]:
                hyper parameters. see VertexPolyGenConfig class for details. 
    """
    
    def __init__(self, model_config):
        super().__init__()
        
        self.tokenizer = DecodeVertexTokenizer(**model_config["tokenizer"])
        self.embedding = VertexDecoderEmbedding(**model_config["embedding"])
        self.reformer = Reformer(**model_config["reformer"])
        self.layernorm = nn.LayerNorm(model_config["embed_dim"])
        self.loss_func = nn.CrossEntropyLoss(ignore_index=model_config["tokenizer"]["pad_id"])
        
        self.apply(init_weights)
    
    def forward(self, tokens, device=None):
        
        """forward function which can be used for both train/predict.
        
        Args
            tokens [dict]: tokenized vertex info.
                `value_tokens` [torch.tensor]:
                        padded (batch, length)-shape long tensor
                        with coord value from 0 to 2^n(bit).
                `coord_type_tokens` [torch.tensor]:
                        padded (batch, length) shape long tensor implies x or y or z.
                `position_tokens` [torch.tensor]:
                        padded (batch, length) shape long tensor
                        representing coord position (NOT sequence position).
                `padding_mask` [torch.tensor]:
                        (batch, length) shape mask implies <pad> tokens.
            device [torch.device]: gpu or not gpu, that's the problem.
            
        
        Returns
            hs [torch.tensor]:
                    hidden states from transformer(reformer) model.
                    this takes (batch, length, embed) shape.
        
        """
        
        hs = self.embedding(tokens)
        hs = self.reformer(
            hs, input_mask=tokens["padding_mask"]
        )
        hs = self.layernorm(hs)
        
        return hs
        
        
    def __call__(self, inputs, device=None):
        
        """Calculate loss while training.
        
        Args
            inputs [dict]: dict containing batched inputs.
                `vertices` [list(torch.tensor)]:
                        variable-length-list of 
                        (length, 3) shaped tensor of quantized-vertices.
            device [torch.device]: gpu or not gpu, that's the problem.
                
        Returns
            outputs [dict]: dict containing calculated variables.
                `loss` [torch.tensor]:
                        calculated scalar-shape loss with backprop info.
                `accuracy` [torch.tensor]:
                        calculated scalar-shape accuracy.
            
        """
        
        tokens = self.tokenizer.tokenize(inputs["vertices"])
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        hs = self.forward(tokens, device=device)
        
        hs = F.linear(hs, self.embedding.value_embed.weight)
        BATCH, LENGTH, EMBED = hs.shape
        hs = hs.reshape(BATCH*LENGTH, EMBED)
        targets = tokens["target_tokens"].reshape(BATCH*LENGTH,)
        
        acc = accuracy(
            hs, targets, ignore_label=self.tokenizer.pad_id, device=device
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
    def predict(self, max_seq_len=2400, device=None):
        """predict function
        
        Args
            max_seq_len[int]: max sequence length to predict.
            device [torch.device]: gpu or not gpu, that's the problem.
            
        Return
            preds [torch.tensor]: predicted (length, ) shape tensor.
        
        """
        
        tokenizer = self.tokenizer
        special_tokens = tokenizer.special_tokens
        
        tokens = tokenizer.get_pred_start()
        tokens = {k: v.to(device) for k, v in tokens.items()}
        preds = []
        pred_idx = 0
        
        while (pred_idx <= max_seq_len-1)\
        and ((len(preds) == 0) or (preds[-1] != special_tokens["eos"]-len(special_tokens))):
            
            if pred_idx >= 1:
                tokens = tokenizer.tokenize([torch.stack(preds)])
                tokens["value_tokens"][:, pred_idx+1] = special_tokens["pad"]
                tokens["padding_mask"][:, pred_idx+1] = True
            
            hs = self.forward(tokens, device=device)

            hs = F.linear(hs[:, pred_idx], self.embedding.value_embed.weight)
            pred = hs.argmax(dim=1) - len(special_tokens)
            preds.append(pred[0])
            pred_idx += 1
            
        preds = torch.stack(preds) + len(special_tokens)
        preds = self.tokenizer.detokenize([preds])[0]
        return preds
