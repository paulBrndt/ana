import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(0xabc0)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# hyperparameters
block_size = 256
eval_iters = 200
C1 = 384
n_head = 6
n_layer = 6
dropout = 0.2

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(C1, head_size, bias=False)
        self.query = nn.Linear(C1, head_size, bias=False)
        self.value = nn.Linear(C1, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, device=device)).to(torch.float32)) # this creates the lower triangle matrix
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C ** 0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
    
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for i in range(num_heads)])
        self.proj = nn.Linear(C1, C1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    

    
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential( # multiplication of 4 comes from the fact that the dimensionality of input is x, but the inner layer dimensionality is 4*x
        nn.Linear(n_embd, 4*n_embd), # linear layer with n_embd input and n_embd output
        nn.ReLU(),# activation function, allows for non linearity (we use ReLU to get over vanishing gradients) -> vanishing gradients is essentially when
        nn.Linear(n_embd * 4, n_embd),    #  the gradients are propagated backward from the output layer to the input layer, they can become very small (vanish) as they pass through many layers.
        nn.Dropout(dropout)          # When the gradients become extremely small, the weights of the early layers are updated only by tiny amounts, if at all.
    )
        
    def forward(self, x):
        return self.net(x)
        
        
        
class Block(nn.Module):
    def __init__(self, n_embd, n_head): # n_embd is the embedding dimension, n_head are the number of heads
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) #self attention
        self.ffwd = FeedForward(n_embd) #feed forward
        self.ln1 = nn.LayerNorm(n_embd) #header normalization layer
        self.ln2 = nn.LayerNorm(n_embd) #header normalization layer
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) #Applies self-attention, normalizes the output with layer normalization, and adds it to the original input.
        x = x + self.ffwd(self.ln2(x)) #Applies the feedforward network, normalizes the output with layer normalization, and adds it to the intermediate result.
        return x
    
    
    
    
# main model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, C1) # table of size vocab_size x C1
        self.position_embedding_table = nn.Embedding(block_size, C1)
        self.blocks = nn.Sequential(*[Block(C1, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(C1)
        self.lm_head = nn.Linear(C1, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C) (batch, time, channel) tensor (4,8,vocab_size)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb # (B,T,C) array, includes both the information about the tokens and their positions in the sequence
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    
    def generate(self,idx, max_new_tokens, verbose=False):
        for i in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                logits, loss = self(idx_cond)
                logits = logits[:, -1, :] # focus on the last time step
                probs = F.softmax(logits, dim=-1) # probabilities
                idx_next = torch.multinomial(probs, num_samples=1) # get the i +1th prediction
                if verbose:
                    print(idx_next[0][0].item())
                idx = torch.cat((idx, idx_next), dim=1)  # concatenate the prediction with the current sequence
        return idx
            
            
            