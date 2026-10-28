import torch
from torch import nn
from transformer import TransformerBlock , LayerNorm
import tiktoken


GPT_124M_CONFIG = {
    "vocabulary_size" : 50257,
    "context_length" : 1024,
    "embedding_dim" : 768,
    "num_attn_heads" : 12,
    "num_layers" : 12,  #Number of transformer blocks
    "dropout_rate" : 0.1,
    "qkv_bias" : False
}

class GPTModel(nn.Module): 
    def __init__(self , cfg):
        super().__init__()
        self.token_embed_layer = nn.Embedding(cfg["vocabulary_size"] , cfg["embedding_dim"])
        self.pos_embed_layer = nn.Embedding(cfg["context_length"] , cfg["embedding_dim"])
        self.dropout = nn.Dropout(cfg["dropout_rate"])

        self.transformer_blks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["num_layers"])])
        self.final_layer_norm = LayerNorm(cfg["embedding_dim"])
        self.linear_output = nn.Linear(cfg["embedding_dim"] , cfg["vocabulary_size"] , bias=False)

    def forward(self , in_tensor):
        batch_size , seq_len = in_tensor.shape
        token_embeddings = self.token_embed_layer(in_tensor)
        position_embeddings = self.pos_embed_layer(
                torch.arange(seq_len , device=in_tensor.device)
        )

        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        x = self.transformer_blks(x)
        x = self.final_layer_norm(x)
        logits = self.linear_output(x)
        
        return logits


def generate_text(model , idx , max_new_tokens , context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[: , -1 , :]
        probability_distrib = torch.softmax(logits , dim=-1)
        idx_next = torch.argmax(probability_distrib , dim=-1 , keepdim=True)
        idx = torch.cat((idx , idx_next) , dim=1)

    return idx


tokenizer = tiktoken.get_encoding("gpt2")
model = GPTModel(GPT_124M_CONFIG)
model.eval()
input_context = input("prompt> ").strip()
encoded = tokenizer.encode(input_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)



out = generate_text(
        model,
        idx = encoded_tensor,
        max_new_tokens = 32,
        context_size = GPT_124M_CONFIG["context_length"]
        )

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
