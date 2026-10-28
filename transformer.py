import torch
from torch import nn

from attention import MultiHeadAttentionFast


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self,x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self , normalized_shape , eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x


class DummyGPTModel(nn.Module):
    def __init__(self , config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config['vocabulary_size'] , config["embedding_dim"])
        self.positional_embeddings = nn.Embedding(config["context_length"] , config["embedding_dim"])
        self.dropout = nn.Dropout(config["dropout_rate"])
        self.trf_blocks = nn.Sequential(
                    *[DummyTransformerBlock(config) for _ in range(config["num_layers"])]
                )
        self.final_norm = DummyLayerNorm(config["embedding_dim"])
        self.out_head = nn.Linear(
                    config['embedding_dim'],
                    config['vocabulary_size'],
                    bias = False
                )

    def forward(self, in_idx):
        batch_size , seq_len = in_idx.shape
        tok_embeds = self.token_embeddings(in_idx)
        pos_embeds = self.positional_embeddings(torch.arange(seq_len , device = in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = torch.sqrt(torch.tensor(2.0 / torch.pi))

    def forward(self , x):
        return 0.5 * x * (1 + torch.tanh(self.factor * (x + 0.44715* torch.pow(x,3)))) 


class LayerNorm(nn.Module):
    def __init__(self,embedding_dim):
        super().__init__()
        self.eps = 1e-5
        #Scale and shift are trainable parameters of the same dimension as the input embeddings.
        # These parameters are applied to the normalized output layer and are adjusted in their training so that
        # They best fit the data they are trained on
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    # This performs the layer normalization
    def forward(self , x):
        mean = x.mean(dim = -1 , keepdim = True)
        variance = x.var(dim = -1 , keepdim = True , unbiased = False)
        normed_x = (x - mean)/torch.sqrt(variance + self.eps)
        return (self.scale * normed_x + self.shift)

class FeedForward(nn.Module):
    def __init__(self , config):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(config["embedding_dim"] , 4* config["embedding_dim"]) ,
                GELU() , 
                nn.Linear(4 * config["embedding_dim"] , config["embedding_dim"])
        )

    def forward(self , x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self , cfg):
        super().__init__()
        self.attention_module = MultiHeadAttentionFast(
                    dim_in = cfg["embedding_dim"] ,
                    dim_out = cfg["embedding_dim"] ,
                    context_length = cfg["context_length"],
                    dropout = cfg["dropout_rate"],
                    num_heads = cfg["num_attn_heads"],
                    qkv_bias = cfg["qkv_bias"]
                )
        self.feed_forward = FeedForward(cfg)
        self.layer_norm1 = LayerNorm(cfg["embedding_dim"])
        self.layer_norm2 = LayerNorm(cfg["embedding_dim"])
        self.dropout_shortcut = nn.Dropout(cfg["dropout_rate"])

    # Implement the transformer
    def forward(self , x):
        normed_embeds = self.layer_norm1(x)
        out1 = self.attention_module(normed_embeds)
        dropout_out = self.dropout_shortcut(out1)
        final_out1 = dropout_out + x

        normed_embeds2 = self.layer_norm2(final_out1)
        ffout = self.feed_forward(normed_embeds2)
        dropout_out = self.dropout_shortcut(ffout)
        final_out = dropout_out + final_out1

        return final_out



