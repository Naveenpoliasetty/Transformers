import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model : int, vocab_size : int ) -> None:  #d_model is the dimension of the embeddings
        super().__init__()
        self.d_model = d_model
        self._vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    
    def forward(self, x):
        return self.embedding(x) * torch.sqrt(self.d_model)
        


class PositionalEncodings(nn.Module):

    def __init__(self, d_model : int, seq_len : int, dropout : float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        #create a matrix of shape(seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        #create a vector of size 512
        position = torch.arange(0,seq_len,dtype=torch.flaot).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        #Apply sin and cosine postion

        pe[:,0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # dimension becomes (1, seq_len, d_model) the extra dimension is added for batch

        self.register_buffer('pe',pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)



class LayerNormalization(nn.Module):
    def __init__(self,features : int, eps : float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) #mulitplicative
        self.bias = nn.Parameter(torch.zeros(features)) #additive

    def forward(self,x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim =True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

class FeedForwardBlock(nn.Module):
    def __init__(self,d_model : int,d_ff : int, dropout : float ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)


    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))



class MultiHeadAttention(nn.Module):
    def __init__(self,d_model : int, dropout : float, h:int ) -> None:
        super().__init__()
        self.h = h
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        assert d_model % h == 0, "D_model is not divisible by Heads"

        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.d_k = d_model // h  #for dinominator in attention formula
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def self_atten(query, key, value, mask, dropout:nn.Linear):
        d_k = key.shape[-1]

        attentionScores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attentionScores.masked_fill(mask == 0, -1e9)
        attentionScores = attentionScores.softmax(dim = -1)
        if dropout is not None:
            attentionScores = dropout(attentionScores)

        return (attentionScores @ value), attentionScores


    def forward(self,q,k,v, mask):
        query = self.w_q(q) #(Batch, seq_len, d_model) -->  (Batch, Seq_len, d_model)
        key = self.w_k(k)  #(Batch, seq_len, d_model) -->  (Batch, Seq_len, d_model)
        value = self.w_v(v)  #(Batch, seq_len, d_model) -->  (Batch, Seq_len, d_model)


        #Batch, seq_len, d_model --> Batch, seq_len, h, d_k, --> Batch, h, seq_len, d_k
        query = query.view(query.shape[0], query.shape[1],self.h,self.d_k).transpose(1,2) 
        key = key.view(key.shape[0],key.shape[1], self.h, self.d_k).transpose(1,2)   
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2) 

        x, attention_scores = MultiHeadAttention.self_atten(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguos().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)



class ResidualConnection(nn.Module):
    def __init__(self, dropout :float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))



class EncoderBlock(nn.Module):
    def __init__(self,self_attention: MultiHeadAttention, feed_forward:FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention = self.self_attention
        self.feed_forward = feed_forward
        self.residual = nn.ModuleList([ResidualConnection(dropout=dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self,layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)
            return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention : MultiHeadAttention, cross_attention:MultiHeadAttention, feed_forward : FeedForwardBlock,dropout:float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual = nn.Module([ResidualConnection(dropout) for _ in range(3)])


    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual[0](x, lambda x : self.self_attention(x,x,x,tgt_mask))
        x = self.residual[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual[2](x,lambda x: self.feed_forward)
        return x
    
class Decoder(nn.Module):
    def __init__(self,layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self,d_model, vocab_size ) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x),dim=-1)
    
class Transformer(nn.Module):
    def __init__(self,encoder:Encoder, decoder:Decoder, src_embed : InputEmbeddings, tgt_embed : InputEmbeddings, src_pos: PositionalEncodings, tgt_pos : PositionalEncodings, projection_layer: ProjectionLayer ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder__output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder__output, src_mask)
    


    def build_transformer(src_vocab_size : int, tgt_vocab_size : int, src_seq_len : int, tgt_seq_len : int, d_model : int = 512, N : int=6, h : int = 8, dropout : float = 0.1, d_ff : int = 2048) -> None:
        #cretating embeddings
        src_embed = InputEmbeddings(d_model, src_vocab_size)
        tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

        #create the positional emebeddings
        src_pos = PositionalEncodings(d_model, src_seq_len, dropout)
        tgt_pos = PositionalEncodings(d_model, tgt_seq_len, dropout)

        #create encoder blocks
        encoder_blocks = []
        for _ in range(N):
            encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block,dropout)
            encoder_blocks.append(encoder_block)

        
        #create the decoder block
        decoder_blocks = []
        for _ in range(N):
            decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
            decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
            decoder_blocks.append(decoder_block)

        
        encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
        decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

        projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

        transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)


        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

        return transformer
    










    


