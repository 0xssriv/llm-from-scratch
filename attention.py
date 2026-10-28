import torch
import torch.nn as nn

# Test inputs below to test the attention mechanism
'''
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts (x^3)
   [0.22, 0.58, 0.33], # with (x^4)
   [0.77, 0.25, 0.10], # one (x^5)
   [0.05, 0.80, 0.55]] # step (x^6)
)
'''
'''
inputs = torch.tensor(
    [
        [0.4 , 0.1 , 0.8],
        [0.5 , 0.8 , 0.6],
        [0.0 , 0.8 , 0.5]
    ]
)
'''

class SelfAttentionV1(nn.Module):
    def __init__(self, dim_in , dim_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(dim_in , dim_out))
        self.W_keys = nn.Parameter(torch.rand(dim_in , dim_out))
        self.W_values = nn.Parameter(torch.rand(dim_in , dim_out))
    def forward(self, x_in_tensor):
        queries = x_in_tensor @ self.W_query
        keys = x_in_tensor @ self.W_keys
        values = x_in_tensor @ self.W_values
        
        scale = keys.shape[-1] ** 0.5
        attn_scores = queries @ (keys.T)
        scaled_attn_scores = attn_scores / scale
        attn_weights = torch.softmax(scaled_attn_scores, dim=-1)

        context_vecs = attn_weights @ values
        return context_vecs


class SelfAttentionV2(nn.Module):
    def __init__(self, dim_in , dim_out, qkv_bias = False):
        super().__init__()
        self.W_query = nn.Linear(dim_in , dim_out, bias=qkv_bias)
        self.W_keys = nn.Linear(dim_in , dim_out, bias=qkv_bias)
        self.W_values = nn.Linear(dim_in , dim_out, bias=qkv_bias)
    def forward(self, x_in_tensor):
        queries = self.W_query(x_in_tensor)
        keys = self.W_keys(x_in_tensor)
        values = self.W_values(x_in_tensor)

        scale = keys.shape[-1] ** 0.5
        raw_attn_scores = queries @ (keys.T)
        #scaled_attn_scores = attn_scores / scale
        #attn_weights = torch.softmax(scaled_attn_scores, dim=-1) #Row is attention weights w.r.to tokens in the sequence cols is the token itself 
        
        #First we must implement masked attention and normalization on the attention scores
        context_length = raw_attn_scores.shape[0]
        mask_tensor = torch.triu(torch.ones(context_length , context_length), diagonal=1)
        masked_attn_scores = raw_attn_scores.masked_fill(mask_tensor.bool() , -torch.inf)
        attn_weights = torch.softmax(masked_attn_scores/scale, dim=1)
        
        #Now we must implement dropout to reduce overtraining
        torch.manual_seed(123)
        dropout = torch.nn.Dropout(0.5) # 50% dropout rate so half the attention weights will be randomly masked and surviving weights are doubled to compensate.
        dropout_attn_weights = dropout(attn_weights)
        print(dropout_attn_weights)
        context_vecs = dropout_attn_weights @ values
        return context_vecs


# A full practical causal attention class which takes in input batches from the encoder.
class CausalAttention(nn.Module):
    def __init__(self, dim_in , dim_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.dim_out = dim_out
        self.W_query = nn.Linear(dim_in, dim_out , bias=qkv_bias) 
        self.W_key = nn.Linear(dim_in, dim_out , bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out , bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # Attention mask tensor is place in register buffer which coordinates between CPU and GPU internally.
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length) , diagonal=1)
        )

    def forward(self,x):
        batchidx , num_tokens , d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        #Calculate raw attention scores. Take transpose of keys with respect to first and second dimensions, across all batches
        raw_attn_scores = queries @ (keys.transpose(1,2))
        #Step-1 Masking attention scores
        scale = keys.shape[-1] ** 0.5
        masked_attn_scores_unnorm = raw_attn_scores.masked_fill_(self.mask.bool()[:num_tokens , :num_tokens] , -torch.inf)
        masked_attn_scores_norm = torch.softmax(masked_attn_scores_unnorm / scale , dim = -1)
        #Step-2 Implement Dropout
        final_attn_weights = self.dropout(masked_attn_scores_norm)
        
        #Finally calculate the context vectors
        context_vectors = final_attn_weights @ values
        return context_vectors


# A basic and naive multi-head attention module, consisting simply of a list of multiple causal attention modules which will operate independently on input embeddings
class MultiHeadAttentionStacked(nn.Module):
    def __init__(self, dim_in , dim_out, context_length, dropout , num_attn_heads , qkv_bias=False):
        super().__init__()

        # Create a pytorch NN module list of causal attention instances, each instance or list element corresponding to an attention head.
        self.heads = nn.ModuleList(
                [CausalAttention(dim_in , dim_out , context_length, dropout, qkv_bias) for _ in range(num_attn_heads)]
        )

    #For the current input, run CausalAttention.forward() on the input from each head and then club the context vectors from each head in a single tensor. 
    def forward(self, x):
        return torch.cat( [attn_head(x) for attn_head in self.heads], dim=-1 )


# An improvement over stacked multi head attention
class MultiHeadAttentionFast(nn.Module):
    def __init__(self, dim_in , dim_out, context_length, dropout, num_heads ,qkv_bias=False):
        super().__init__()
        # We will divide the output dimensions by number of heads. So the output vector you see is actually a result from multiple heads.
        assert (dim_out % num_heads) == 0 , "Error! Output dimensions must be divisible ny number of heads"

        self.dim_out = dim_out
        self.num_heads = num_heads
        self.dim_head = dim_out // num_heads
        self.W_query = nn.Linear(dim_in, dim_out , bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out , bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out , bias=qkv_bias)
        #Linear layer to combine head outputs
        self.out_proj = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(dropout)
        # Attention mask tensor is place in register buffer which coordinates between CPU and GPU internally.
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length) , diagonal=1)
        )

    def forward(self,x):
        batchidx , num_tokens , d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(batchidx , num_tokens , self.num_heads , self.dim_head)
        queries = queries.view(batchidx , num_tokens , self.num_heads , self.dim_head)
        values = values.view(batchidx , num_tokens , self.num_heads , self.dim_head)

        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        #Calculate raw attention scores. Take transpose of keys with respect to first and second dimensions, across all batches
        raw_attn_scores = queries @ (keys.transpose(2,3))
        #Step-1 Masking attention scores
        scale = keys.shape[-1] ** 0.5
        masked_attn_scores_unnorm = raw_attn_scores.masked_fill_(self.mask.bool()[:num_tokens , :num_tokens] , -torch.inf)
        masked_attn_scores_norm = torch.softmax(masked_attn_scores_unnorm / scale , dim = -1)
        #Step-2 Implement Dropout
        final_attn_weights = self.dropout(masked_attn_scores_norm)

        #Finally calculate the context vectors
        context_vectors = (final_attn_weights @ values).transpose(1,2)
        context_vectors = context_vectors.contiguous().view(batchidx , num_tokens, self.dim_out)
        context_vectors = self.out_proj(context_vectors)

        return context_vectors

#torch.manual_seed(789)
#self_attn = SelfAttentionV2(3,2)
#print(self_attn(inputs))

