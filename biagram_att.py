
import torch
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import numpy as np


batch_size=64
block_size=256
max_iters=3
eval_interval=300
learning_rate=3e-4
device='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200
n_embd=384
n_head=6
n_layers=6
dropout=0.2

torch.manual_seed(1337)

with open("names.txt",'r',encoding='utf-8') as f :
    words=f.read()

chars=sorted(list(set(words)))
stoi={s: i for i,s in enumerate(chars)}
itos={i:s for s,i in stoi.items()}
encode= lambda s : [stoi[c] for c in s] #-->int
decode=lambda i :  "".join([itos[x] for x in i  ])


data=torch.tensor(encode(words),dtype=torch.long)
vocab_size=len(chars)
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]


#Loading data

def get_batch(split):

    data=train_data if split=='train' else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,))

    x=torch.stack([data[i:i+block_size] for i in ix ])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix ]) ##Next character in a sequence
    x,y=x.to(device),y.to(device)

    return x,y



@torch.no_grad()
def estimate_loss():
    out={}
    m.eval()
    for split in ["train" ,"val"] :
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters) : 
            X,Y=get_batch(split)
            logits,loss=m(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    m.train()
    return out


class Head(nn.Module) :
    def __init__(self,head_size) -> None:
        super().__init__()
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)
        self.dropout=nn.Dropout(dropout)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))


    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)
        
        wei=q @ k.transpose(-2,-1)*C**-0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0,value=float('-inf'))
        wei=F.softmax(wei,dim=-1)
        wei=self.dropout(wei)
        out=wei@v

        return out




class MultiHeadAttention(nn.Module):

    def __init__(self,num_head,head_size) :
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj=nn.Linear(n_embd,n_embd) #projection onto the residual pathway
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x):
        out= torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.proj(out)
        out=self.dropout(out)   
        return out





class FFN(nn.Module):
    def __init__(self,n_embd) :
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout)
            
        )
    
    def forward(self,x):
        
        return self.net(x)


class Block(nn.Module):
    ##Transformer block

    def __init__(self,n_embd,n_head) :
        super().__init__()
        #n_embd dimension of embedding, n_head number of Head

        head_size=n_embd//n_head
        self.sa_heads=MultiHeadAttention(n_head,head_size)
        self.ffn=FFN(n_embd)
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)
    
    def forward(self,x):
        x=x+self.sa_heads(self.ln1(x))
        x=x+self.ffn(self.ln2(x))
        return x




class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
        # self.sa_heads=MultiHeadAttention(1,n_embd)
        # self.block=nn.Sequential(
        #     Block(n_embd,n_head=4),
        #     Block(n_embd,n_head=4),
        #     Block(n_embd,n_head=4),
        # )

        self.block=nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layers)])
        self.lm_head=nn.Linear(n_embd,vocab_size)
        self.ffn=FFN(n_embd)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        B,T=idx.shape
        token_emb = self.token_embedding_table(idx) # (B,T,C)
        position_emb=self.position_embedding_table(torch.arange(T,device=device))
        x=token_emb+position_emb

        # x=self.sa_heads(x)
        x=self.block(x)
        x=self.ffn(x)
        
        logits=self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            
            idx_cond=idx[:,-block_size:]

            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel()
m=m.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)



for iter in range(max_iters): 
    
    
    if iter%eval_interval==0 :
        losses=estimate_loss()
        print(f'step {iter} : train loss {losses["train"] : 4f}, val loss {losses["val"] :4f}')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()



context=torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(m.generate(context,max_new_tokens=500)[0].tolist()))