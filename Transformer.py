import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import math 

batch_size=32
block_size=8
max_iters=300
eval_interval=300
learning_rate=3e-4
device='cuda' if torch.cuda.is_available() else 'cpu'
dropout=0.2



class InputEmbedding(nn.Module):
    def __init__(self,d_model, vocab_size):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)
    
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)
    


class PositionalEmbedding(nn.Module) :
    def __init__(self,d_model,seq_len) -> None:
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        pe=torch.zeros(seq_len,d_model)
        position=torch.arange(seq_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(-torch.arange(0,d_model,2)*math.log(10000)/d_model)
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        pe=pe.unsqueeze(dim=0) #batch dimension
        self.register_buffer("pe",pe)

    def forward(self,x):
        out=x+(self.pe[:,:x.shape[1],:]).requires_grad_(False)
        out=self.dropout(out)

        return out
        


class FeedForward(nn.Module):
    def __init__(self, d_model,d_ff) -> None:
        #d_ff size of the hidden state
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff,d_model)
        )
    def forward(self, x):
        #(Batch, seq_len,d_model)-> (Batch,seq_len,d_ff) -> (Batch_size,seq_len,d_model)
        
        out=self.net(x)
        return out




class MultiHeadAttention(nn.Module):
    def __init__(self, d_model,num_head, head_size) -> None:
        super().__init__()

        self.h=num_head
        self.d_model=d_model
        self.dh=head_size

        self.q=nn.Linear(d_model,d_model)
        self.k=nn.Linear(d_model,d_model)
        self.v=nn.Linear(d_model,d_model)         
        self.dropout=nn.Dropout(dropout)

       
        self.proj=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)

    
    @staticmethod
    def attention(query,key,value,mask,dropout : nn.Dropout):
        dhead=query.shape[-1]
        

        attention_scores=query@key.transpose(-2,-1)*dhead**-0.5
        if  mask is not None: 
            attention_scores=attention_scores.masked_fill(mask==0,value=float('-inf'))    

        attention=F.softmax(attention_scores,dim=-1)
        if dropout is not None : 
            attention_scores=dropout(attention_scores)

        attention=attention_scores@value
        return attention,attention_scores

        
    
    def forward(self,query,key,value,mask):

      
        q=self.q(query)
        k=self.k(key) 
        v=self.v(value)

        B,seq_len,embd=q.shape

        # (Batch,seq_len,d_model)->(Batch,seq_len,n_head,head_size)->(Batch,n_head,seq_len,head_size)
        q=q.view(B,seq_len,self.h,self.dh).transpose(1,2)
        k=k.view(B,seq_len,self.h,self.dh).transpose(1,2)
        v=v.view(B,seq_len,self.h,self.dh).transpose(1,2)

        x,attention_score=MultiHeadAttention.attention(q,k,v,mask,self.dropout)
        x=x.transpose(1,2).contiguous().view(B,seq_len,self.h*self.dh)#(Batch,n_head,seq_len,head_size)->(Batch,seq_len,n_head,head_size) ->(Batch,seq_len,d_model)

        out=self.proj(x)

            # (Batch,seq_len,d_model)->(Batch,seq_len,d_model)
        return out


class Encoder(nn.Module):
    def __init__(self, d_model, n_head,d_ff) -> None:
        super().__init__()
        
        assert d_model % n_head ==0
        
        self.sa_heads=MultiHeadAttention(d_model=d_model,num_head=n_head,head_size=d_model//n_head)
        self.mlp=FeedForward(d_model,d_ff)
        self.ln1=nn.LayerNorm(d_model)
        self.ln2=nn.LayerNorm(d_model)
    
    def forward(self,x,mask):
        _x=self.ln1(x)
        out=self.sa_heads(_x,_x,_x,mask)
        out=out + x
        out=self.mlp(out) + out

        return out


class EncoderBlock(nn.Module) :
    def __init__(self,n_layers,d_model,n_head) -> None:
        super().__init__()
        self.encoder_block=nn.ModuleList([Encoder(d_model=d_model,n_head=n_head,d_ff=4*d_model) for _ in range(n_layers)])
        self.ln=nn.LayerNorm(d_model)
    

    def forward(self,x,src_mask):
        for layers in self.encoder_block :
            out=layers(x,src_mask)
        out=self.ln(out)

        return out



class Decoder(nn.Module):
    def __init__(self ,d_model,n_head):
        super().__init__()
        self.sa_heads=MultiHeadAttention(d_model=d_model,num_head=n_head,head_size=d_model//n_head)
        self.cross_att=MultiHeadAttention(d_model=d_model,num_head=n_head,head_size=d_model//n_head)

        self.mlp=FeedForward(d_model,d_ff=4*d_model)
        self.ln1=nn.LayerNorm(d_model)
        self.ln3=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,encoder_output,x,mask_src,mask_tgt):
        _x=self.ln1(x)
        out=self.sa_heads(_x,_x,_x,mask_tgt) #causal mask -attention
        out=out + x

        if encoder_output is  not None : 
            out=self.cross_att(out,encoder_output,encoder_output,mask_src)
            out=self.dropout(out)
        
        out=self.mlp(out) + out

        return out
    
class DecoderBlock(nn.Module) :
    def __init__(self,n_layers,d_model,n_head) -> None:
        super().__init__()
        self.decoder_block=nn.ModuleList([Decoder(d_model=d_model,n_head=n_head) for _ in range(n_layers)])
        self.ln=nn.LayerNorm(d_model)

    

    def forward(self,encoder_output,x,src_mask,tgt_mask):
        _x=x
        for layer in self.decoder_block :
            out=layer(_x,encoder_output,src_mask,tgt_mask)

        out=self.ln(out)

        return out


class ProjectionLayer(nn.Module):
    def __init__(self,d_model,vocab_size) -> None:
        super().__init__()
        self.proj=nn.Linear(d_model,vocab_size)
    
    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim=-1,dtype=torch.int64)



class Transformer(nn.Module):
    def __init__(self,encoder : EncoderBlock, decoder:DecoderBlock, src_embed:InputEmbedding, tgt_embed :InputEmbedding ,src_pos:PositionalEmbedding, target_pos:PositionalEmbedding, proj_layer:ProjectionLayer) -> None:

        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.src_pos=src_pos
        self.tgt_embed=tgt_embed
        self.tgt_pos=target_pos
        self.proj=proj_layer
    

    def encode(self,src,src_mask):
        src=self.src_embed(src)
        src=self.src_pos(src)
        return self.encoder(src,src_mask)

    def decode(self,encoder_output,src_mask,tgt, tgt_mask):
        tgt=self.tgt_embed(tgt)
        tgt=self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    

    def project(self,x) :
        return self.proj(x)




def build_transformers(src_vocab_size,tgt_vocab_size,src_seq_len,tgt_seq_len,d_model=512,N=6,h=8, d_ff=2048):
    src_embed=InputEmbedding(d_model,src_vocab_size)
    target_embed=InputEmbedding(d_model,tgt_vocab_size)

    #Posotional embedding
    src_pos=PositionalEmbedding(d_model,src_seq_len)
    tgt_pos=PositionalEmbedding(d_model,tgt_seq_len)


    ##Encoder/ Deocder Block
    encoder_blocks=EncoderBlock(n_layers=N,d_model=d_model,n_head=h)
    decoder_blocks=DecoderBlock(n_layers=N,d_model=d_model,n_head=h)


    projection_layer=ProjectionLayer(d_model,tgt_vocab_size)
    transformer=Transformer(encoder_blocks,decoder_blocks,src_embed,target_embed,src_pos,tgt_pos,proj_layer=projection_layer)

    #Initialize the weightsg
    for p in transformer.parameters() :
        if p.dim()>1 :
            nn.init.xavier_uniform_(p)
    

    return transformer





