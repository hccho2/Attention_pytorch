'''
https://bastings.github.io/annotated_encoder_decoder/
https://sigmoidal.io/implementing-additive-attention-in-pytorch/

1. data가 적은 경우: Location Sensitive Attention은 overfitting인 되는데,
Bahdanau Attention, Bahdanau Monotonic Attention은 overfitting도 안됨.

--> TTS에선 입력이 text이다. monotonic attention이 적합하다.
--> STT에서는 입력되는 음성 feature의 길이가 길다. 입력의 길이가 길기 때문에, monotonic attention이 적합하지 않을 수도 있다.
--> 입력 길이를 target text 길이 많큼 줄이면 어떻게 될까?



2. data가 많은 경우는 어떻게 될까??? 


'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class Attention(nn.Module):
    """
    Location-based
    """
    def __init__(self, dec_dim, enc_dim, conv_dim, attn_dim, smoothing=False):
        super(Attention, self).__init__()
        self.dec_dim = dec_dim
        self.enc_dim = enc_dim
        self.conv_dim = conv_dim  # 사용 안함.
        self.attn_dim = attn_dim  # decoder hidden dim
        self.smoothing= smoothing
        self.conv = nn.Conv1d(in_channels=1, out_channels=self.attn_dim, kernel_size=3, padding=1)  # same padding, Time 축에 대한 convolution  

        self.W = nn.Linear(self.dec_dim, self.attn_dim, bias=False)
        self.V = nn.Linear(self.enc_dim, self.attn_dim, bias=False)

        self.fc = nn.Linear(attn_dim, 1, bias=True)
        self.b = nn.Parameter(torch.rand(attn_dim))  # trainable bias 

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, queries, values, last_attn):
        """
        param:quries: Decoder hidden states, Shape=(B,1,dec_D) <---- 넘겨 받을 때, unsqueeze해서 받았다.
        param:values: Encoder outputs, Shape=(B,enc_T,enc_D)  ---> key
        param:last_attn: Attention weight of previous time step, Shape=(batch, enc_T)
        """
        batch_size = queries.size(0)
        dec_feat_dim = queries.size(2)
        enc_feat_len = values.size(1)

        # conv_attn = (B, enc_T, conv_D)
        conv_attn = torch.transpose(self.conv(last_attn.unsqueeze(dim=1)), 1, 2)  # (N,C=1,enc_T) ---> (N,dec_D,enc_T) ---> transpose --> (N,T,attn_dim)

        # (B, enc_T)
        score =  self.fc(self.tanh(
         self.W(queries) + self.V(values) + conv_attn + self.b
        )).squeeze(dim=-1)  # (N,enc_T,1) --> squeeze(N,enc_T)


        if self.mask is not None:
            score.data.masked_fill_(self.mask, -float('inf'))

        # attn_weight : (B, enc_T)
        if self.smoothing:
            score = torch.sigmoid(score)
            attn_weight = torch.div(score, score.sum(dim=-1).unsqueeze(dim=-1))
        else:
            attn_weight = self.softmax(score) 

        # (B, 1, enc_T) * (B, enc_T, enc_D) -> (B, 1, enc_D) 
        context = torch.bmm(attn_weight.unsqueeze(dim=1), values)

        return context, attn_weight
class LacationSensitiveAttention(nn.Module):
    """
    Attention class를 좀 더 효율적이게 수정. mask도 추가.
    """
    def __init__(self, encoder_dim, decoder_dim, conv_dim, num_units, smoothing=False):
        super(LacationSensitiveAttention, self).__init__()
        self.dec_dim = decoder_dim
        self.enc_dim = encoder_dim
        self.conv_dim = conv_dim  # 주로 1.
        self.num_units = num_units  # attention dim: 주로 decoder hidden dim을 사용하지만, 그렇지 않아도 된다.
        self.smoothing= smoothing
        self.conv = nn.Conv1d(in_channels=conv_dim, out_channels=self.num_units, kernel_size=3, padding=1)  # same padding, Time 축에 대한 convolution  

        self.query_layer = nn.Linear(self.dec_dim, self.num_units, bias=False)
        self.key_layer = nn.Linear(self.enc_dim, self.num_units, bias=False)

        self.v = nn.Linear(num_units, 1, bias=True)
        self.b = nn.Parameter(torch.rand(num_units))  # trainable bias 

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def set_memory(self,encoder_output,lengths=None):
        # encoder_output: (N,Te,encoder_dim)
        self.memory = encoder_output
        self.proj_key = self.key_layer(encoder_output)
        if lengths is not None:
            maxlen = encoder_output.size(1) # Te: encoder_length
            self.mask = torch.arange(maxlen)[None, :].to(lengths.device) >= lengths[:, None]  # mask 할 부분이 True  --> (N,Te)
        else:
            self.mask = None


    def forward(self, queries,previous_alignments):
        """
        param:quries: Decoder hidden states, Shape=(B,1,dec_D) <---- 넘겨 받을 때, unsqueeze해서 받았다.
        param:values: Encoder outputs, Shape=(B,enc_T,enc_D)  ---> key
        param:previous_alignments: Attention weight of previous time step, Shape=(batch, enc_T)
        """

        # conv_attn = (B, enc_T, conv_D)
        conv_attn = torch.transpose(self.conv(previous_alignments.unsqueeze(dim=1)), 1, 2)  # (N,C=1,enc_T) ---> (N,dec_D,enc_T) ---> transpose --> (N,T,attn_dim)

        # (B, enc_T)
        
        weights = self.query_layer(queries) + self.proj_key + conv_attn + self.b
        
        score =  self.v(self.tanh(weights)).squeeze(dim=-1)  # (N,enc_T,1) --> squeeze(N,enc_T)


        if self.mask is not None:
            score.data.masked_fill_(self.mask, -float('inf'))

        # attn_weight : (B, enc_T)
        if self.smoothing:
            score = torch.sigmoid(score)
            attn_weight = torch.div(score, score.sum(dim=-1).unsqueeze(dim=-1))
        else:
            attn_weight = self.softmax(score) 

        # (B, 1, enc_T) * (B, enc_T, enc_D) -> (B, 1, enc_D) 
        context = torch.bmm(attn_weight.unsqueeze(dim=1), self.memory)

        return context, attn_weight


class DotProductAttention(nn.Module):
    '''
        - dot product attention
        - encorder_dim = decoder_dim 인 경우에 사용 가능
    '''
    def __init__(self,dim=None,scaled=False):          
        super().__init__()
        
        self.scaled = scaled
        if self.scaled:
            self.dim = dim
            self.scale_factor = np.sqrt(dim)                      
        
    def set_memory(self,encoder_output,lengths=None):
        # encoder_output: (N,Te,D)
        self.memory = encoder_output
        if lengths is not None:
            maxlen = encoder_output.size(1) # Te: encoder_length
            self.mask = torch.arange(maxlen)[None, :].to(lengths.device) >= lengths[:, None]  # mask 할 부분이 True  --> (N,Te)
        else:
            self.mask = None

    def forward(self,queries,previous_alignments=None):
        '''
            query: [N,Td,D]  <---- Td = 1일 필요가 없다.
        '''
        weights = torch.bmm(queries,self.memory.transpose(1,2))  # (N,Td,D) * (N,Te,D).T ===> (N,Td,Te)

        if self.scaled:
            weights = torch.div(weights,self.scale_factor)

        if self.mask is not None:
            weights.data.masked_fill_(self.mask.unsqueeze(dim=1),-float('inf'))      

        alignments = F.softmax(weights, dim=-1) # [N,Td,Te]
        
        contexts = torch.bmm(alignments, self.memory)  # (N,Td,Te)
        
        return contexts, alignments



class LuongAttention(nn.Module):
    '''
        - general dot product attention
    '''
    def __init__(self, encoder_dim, decoder_dim):          
        super().__init__()                  
        self.va = torch.nn.Linear(encoder_dim, decoder_dim,bias=False)          
        
    def set_memory(self,encoder_output,lengths=None):
        # encoder_output: (N,Te,encoder_dim)
        self.memory = encoder_output
        if lengths is not None:
            maxlen = encoder_output.size(1) # Te: encoder_length
            self.mask = torch.arange(maxlen)[None, :].to(lengths.device) >= lengths[:, None]  # mask 할 부분이 True  --> (N,Te)
        else:
            self.mask = None

    def forward(self,queries,previous_alignments=None):
        '''
            query: [N,1,decoder_dim]  <---- dedocer hidden   
        '''
        weights = torch.bmm(self.va(self.memory), queries.transpose(1, 2)).squeeze(dim=-1)   # (N,Te,decoder_dim) * (N,1,decoder_dim) ===> (N,Te)

        if self.mask is not None:
            weights.data.masked_fill_(self.mask,-float('inf'))      

        alignments = F.softmax(weights, dim=-1) # [N,Te]
        
        contexts = torch.bmm(alignments.unsqueeze(dim=1), self.memory).squeeze(dim=1)
        
        return contexts, alignments



class BahdanauAttention(nn.Module):
    '''
        - tensorflow의 BahdanauAttention과 유사하게 구현.
    
    '''
    def __init__(self, encoder_dim, decoder_dim, num_units, normalize=False):          
        super().__init__()          

        self.encoder_dim = encoder_dim          
        self.decoder_dim = decoder_dim
        self.num_units = num_units  # attention dim
        self.v = torch.nn.Parameter(torch.rand(self.num_units))          
        self.query_layer = torch.nn.Linear(self.decoder_dim, self.num_units,bias=False)          
        self.key_layer = torch.nn.Linear(self.encoder_dim, self.num_units,bias=False)          
        
        self.normalize = normalize
        
        if normalize:
            self.attention_g = torch.nn.Parameter( torch.tensor(1.0/self.num_units).sqrt())
            self.attention_b = torch.nn.Parameter(torch.zeros(self.num_units))

    def set_memory(self,encoder_output,lengths=None):
        # encoder_output: (N,Te,encoder_dim)
        # mask(boolean) 할 부분이 True가 될 수 있도록 만든다.
        self.memory = encoder_output
        self.proj_key = self.key_layer(encoder_output)
        if lengths is not None:
            maxlen = encoder_output.size(1) # Te: encoder_length
            self.mask = torch.arange(maxlen)[None, :].to(lengths.device) >= lengths[:, None]  # mask 할 부분이 True  --> (N,Te)
            
        else:
            self.mask = None

    def forward(self,queries,previous_alignments=None):
        '''
            query: [N,1,decoder_dim]  <---- dedocer hidden   
        '''
        
        #queries = queries.unsqueeze(1)  # broadcasting을 위해 reshape   (N,1,decoder_dim]
        
        if self.normalize:
            weights = self.query_layer(queries) + self.proj_key + self.attention_b
            weights = self.attention_g * (torch.tanh(weights) @ self.v) / self.v.norm()  # self.v.square().sum().sqrt()
        else:
            weights = self.query_layer(queries) + self.proj_key # --> [N,Te, num_units]          
            weights = torch.tanh(weights) @ self.v # [Te, num_units]@ [num_units]  ==>  [Te]
        
        
        if self.mask is not None:
            weights.data.masked_fill_(self.mask,-float('inf'))      

        alignments = F.softmax(weights, dim=-1) # [N,Te]
        
        
        contexts = torch.bmm(alignments.unsqueeze(dim=1), self.memory).squeeze(dim=1)
        return contexts, alignments


def safe_cumprod(x, exclusive=False):
    # x: (N,T)
    """Numerically stable cumulative product by cumulative sum in log-space"""
    bsz = x.size(0)
    logsum = torch.cumsum(torch.log(torch.clamp(x, min=1e-20, max=1)), dim=1)
    if exclusive:
        logsum = torch.cat([torch.zeros(bsz, 1).to(logsum), logsum], dim=1)[:, :-1]
    return torch.exp(logsum)

def monotonic_attention(p_choose_i, previous_attention):
    # Tensorflow에 구현되어 있는 mode = 'parallel' 버전.
    # p_choose_i: (N,T)
    cumprod_1mp_choose_i = safe_cumprod(1 - p_choose_i, exclusive=True)
    attention = p_choose_i * cumprod_1mp_choose_i * torch.cumsum(
        previous_attention /
        # Clip cumprod_1mp to avoid divide-by-zero
        torch.clamp(cumprod_1mp_choose_i, 1e-10, 1.),
        dim=1)
    return attention
    
    
    
    
class BahdanauMonotonicAttention(nn.Module):
    '''
        - tensorflow의 BahdanauAttention과 유사하게 구현.
    
    '''
    def __init__(self, encoder_dim, decoder_dim, num_units, normalize=False):          
        super().__init__()          

        self.encoder_dim = encoder_dim          
        self.decoder_dim = decoder_dim
        self.num_units = num_units    
        self.v = torch.nn.Parameter(torch.rand(self.num_units))          
        self.query_layer = torch.nn.Linear(self.decoder_dim, self.num_units,bias=False)          
        self.key_layer = torch.nn.Linear(self.encoder_dim, self.num_units,bias=False)          
        
        self.normalize = normalize
        
        if normalize:
            self.attention_g = torch.nn.Parameter( torch.tensor(1.0/self.num_units).sqrt())
            self.attention_b = torch.nn.Parameter(torch.zeros(self.num_units))
            self.score_bias = torch.nn.Parameter( torch.tensor(0.))

    def set_memory(self,encoder_output,lengths=None):
        # encoder_output: (N,Te,encoder_dim)
        self.memory = encoder_output
        self.proj_key = self.key_layer(encoder_output)
        if lengths is not None:
            maxlen = encoder_output.size(1) # Te: encoder_length
            self.mask = torch.arange(maxlen)[None, :].to(lengths.device) >= lengths[:, None]  # mask 할 부분이 True  --> (N,Te)
            
        else:
            self.mask = None

    def forward(self,queries,previous_alignments):
        '''
            query: [N,1,decoder_dim]  <---- dedocer hidden   
        '''
        
        #queries = queries.unsqueeze(1)  # broadcasting을 위해 reshape   (N,1,decoder_dim]
        
        if self.normalize:
            weights = self.query_layer(queries) + self.proj_key + self.attention_b
            weights = self.attention_g * (torch.tanh(weights) @ self.v) / self.v.norm()  # self.v.square().sum().sqrt()
            weights = weights + self.score_bias
        else:
            weights = self.query_layer(queries) + self.proj_key # --> [N,Te, num_units]          
            weights = torch.tanh(weights) @ self.v # [N,Te, num_units]@ [num_units]  ==>  [N,Te]
        
        
        if self.mask is not None:
            weights.data.masked_fill_(self.mask,-float('inf'))      
        
        #weights = torch.randn(self.memory.size(0),self.memory.size(1))
        alignments = monotonic_attention(torch.sigmoid(weights),previous_alignments) # [N,Te]
        
        
        contexts = torch.bmm(alignments.unsqueeze(dim=1), self.memory).squeeze(dim=1)
        return contexts, alignments






class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model,d_model)
        self.wk = nn.Linear(d_model,d_model)
        self.wv = nn.Linear(d_model,d_model)

        self.dense = nn.Linear(d_model,d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        x = x.permute(0,2,1,3).contiguous().view(batch_size*self.num_heads,-1,self.depth)
        return x

    def forward(self,q,k,v, key_padding_mask=None, attn_mask=None):
        # key_padding_mask: True/False값.
        # attn_mask: 0 또는 -inf
        
        
        # key_padding_mask: padding mask, (N,seq_len_k)
        # attn_mask: causal mask, (seq_len_q,seq_len_k)  --> batch_size 무관
        batch_size = q.size(0)

        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size*num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size*num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size*num_heads, seq_len_v, depth)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(batch_size, 1, 1, -1). \
                expand(-1, self.num_heads, -1, -1).reshape(batch_size * self.num_heads, 1, -1)  # (N*num_heads,1,seq_len_k)
        
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)   # braodcasting
            else:
                attn_mask = attn_mask.unsqueeze(0)
                attn_mask.masked_fill(key_padding_mask, float("-inf"))


        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask


        # attn_mask shape
        #     - 입력된 attn_mask = None ---> (N*num_heads,1,seq_len_k)
        #     - 입력된 attn_mask = None이 아니면 ---> (N*num_heads,seq_len_q,seq_len_k)


        # scaled_attention.shape == (batch_size*num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size*num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, attn_mask)


        scaled_attention = scaled_attention.view(batch_size, self.num_heads, -1, self.depth).permute(0,2,1,3)  # (batch_size, seq_len_q, d_model)
        concat_attention = scaled_attention.contiguous().view(batch_size,-1,self.d_model)


        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights  # (N,seq_len_q,D), (N,num_heads,seq_len_q,seq_len_k)  seq_len_k = seq_len_v


    def scaled_dot_product_attention(self,q, k, v, mask=None):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.
    
        Args:
          q: query shape == (..., seq_len_q, depth)
          k: key shape == (..., seq_len_k, depth)
          v: value shape == (..., seq_len_v, depth_v)
          mask: Boolean tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k). Defaults to None.
    
        Returns:
          output, attention_weights
        """
    
        matmul_qk = torch.bmm(q,k.transpose(-2,-1))  # q @ k.transose(-1,-2)  # (..., seq_len_q, seq_len_k)
    
        # scale matmul_qk
        dk = np.sqrt(k.size(-1))
        scaled_attention_logits = matmul_qk / dk  # (N*num_heads,seq_len_q, seq_len_k)
    
        # add the mask to the scaled tensor.
        if mask is not None:
            # scaled_attention_logits: (N*num_heads,seq_len_q, seq_len_k)
            # mask: (N*num_heads,1, seq_len_k) 또는 (N*num_heads,seq_len_q, seq_len_k)
            scaled_attention_logits += mask
    
        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # (..., seq_len_q, seq_len_k)  = (..., Td, Te)
    
        output = torch.bmm(attention_weights, v)  # (..., seq_len_q, depth_v) = (..., Td, D)
    
        return output, attention_weights

def test_attention1():
    # BahdanauAttention
    batch_size=2
    encoder_length = torch.tensor([4,6])
    encoder_dim = 3
    decoder_dim = 4
    num_units = 4
    
    query = torch.randn(batch_size,1,decoder_dim)
    memory = torch.randn(batch_size,6,encoder_dim)
    
    
    attention = BahdanauAttention(encoder_dim,decoder_dim,num_units,normalize=True)
    
    
    attention.set_memory(memory,encoder_length)
    contexts,alignments = attention(query)

    print(f'contexts: {contexts}')
    print(f'alignments: {alignments}')

def test_attention2():
    # BahdanauMonotonicAttention
    batch_size=2
    encoder_length = torch.tensor([4,6])
    
    encoder_dim = 3
    decoder_dim = 4
    num_units = 4
    
    
    memory = torch.randn(batch_size,6,encoder_dim)
    
    
    attention = BahdanauMonotonicAttention(encoder_dim,decoder_dim,num_units,normalize=False)
    
    
    attention.set_memory(memory,encoder_length)
    max_encoder_length = memory.size(1)
    
    alignments_all = []
    
    
    previous_alignments = F.one_hot(torch.zeros(batch_size,dtype=torch.int64),max_encoder_length)
    for i in range(7): # decoder length 만큼 loop
        query = torch.randn(batch_size, 1, decoder_dim)
        
        contexts,alignments = attention(query,previous_alignments)
        
        alignments_all.append(alignments.detach().numpy())
        previous_alignments = alignments


    alignments_all = np.stack(alignments_all,axis=1)
    plt.plot(alignments_all[1].T)
    plt.legend(range(6))
    plt.show()




def test_attention3():
    # Luong Attention, DotProductAttention(길이 1인 decoder query)
    batch_size=2
    encoder_length = torch.tensor([4,6])
    
    encoder_dim = 3
    decoder_dim = 4

    
    
    memory = torch.randn(batch_size,6,encoder_dim)
    
    mode = 2
    if mode ==1:
        attention = LuongAttention(encoder_dim,decoder_dim)
    else:
        decoder_dim = encoder_dim
        attention = DotProductAttention(encoder_dim,scaled=True)


    attention.set_memory(memory,encoder_length)
    
    alignments_all = []
    
    
    for i in range(7): # decoder length 만큼 loop
        query = torch.randn(batch_size, 1, decoder_dim)
        
        contexts,alignments = attention(query)
        
        if mode != 1:
            alignments = alignments.squeeze(dim=1)
        alignments_all.append(alignments.detach().numpy())



    alignments_all = np.stack(alignments_all,axis=1)
    plt.plot(alignments_all[1].T)
    plt.legend(range(7))
    plt.show()



def test_attention4():
    # DotProductAttention(query 길이 1 이상인 경우)
    batch_size=2
    encoder_length = torch.tensor([4,6])
    
    D = 3
    scaled = True
    
    memory = torch.randn(batch_size,6,D)
    
    attention = DotProductAttention(D,scaled)

    alignments_all = []
    
    
    attention.set_memory(memory,encoder_length)
    query = torch.randn(batch_size, 7, D)
    
    contexts,alignments = attention(query)
    
    plt.plot(alignments[1].T)
    plt.legend(range(7))
    plt.show()

def test_attention5():
    # MultiHeadAttention
    
    def generate_square_subsequent_mask(length, boolean_flag=False):
        # causal padding
        if boolean_flag:
            mask = (torch.triu(torch.ones(length, length)) == 0).transpose(0, 1)
        else:
            mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    
    
    batch_size=2
    decoder_length = torch.tensor([4,7])
    decoder_maxlen = decoder_length.max().item()
    

    encoder_length = torch.tensor([3,6])
    encoder_maxlen = encoder_length.max().item()

    
    D = 64
    n_head = 8
    
    memory = torch.randn(batch_size,encoder_maxlen,D)

    query = torch.randn(batch_size,decoder_maxlen,D)

    attention = MultiHeadAttention(D,n_head)

    alignments_all = []
    
    
    query = torch.randn(batch_size, decoder_maxlen, D)
    
    # padding_mask: True/False값.
    # attn_mask: 0 또는 -inf  ---> boolean 값이어도 된다.
    encoder_padding_mask = torch.arange(encoder_maxlen)[None, :].to(encoder_length.device) >= encoder_length[:, None] # (N,encoder_maxlen)
    decoder_padding_mask = torch.arange(decoder_maxlen)[None, :].to(decoder_length.device) >= decoder_length[:, None] # (N,decoder_maxlen)
    
    
    attn_mask = generate_square_subsequent_mask(decoder_maxlen,boolean_flag=True)  # 0 또는 -inf, causal padding (Td,Td)
    
    contexts,alignments = attention(query,query,query,decoder_padding_mask,attn_mask)  # decoder self attention
    print(contexts.shape, alignments.shape)
    
    contexts,alignments = attention(query,memory,memory,encoder_padding_mask)  # cross attention
    print(contexts.shape, alignments.shape)
    

    contexts,alignments = attention(query,query,query,decoder_padding_mask)  # decoder self attention
    print(contexts.shape, alignments.shape)


   
    plt.imshow(alignments[0].detach().numpy())
    plt.show()


if __name__ ==  '__main__':
    #test_attention1()
    #test_attention2()
    #test_attention3()
    #test_attention4()
    test_attention5()






# class BahdanauAttention_(nn.Module):
#     # https://bastings.github.io/annotated_encoder_decoder/
#     """Implements Bahdanau (MLP) attention"""
#
#     def __init__(self, hidden_size, key_size=None, query_size=None):
#         super(BahdanauAttention_, self).__init__()
#
#         # We assume a bi-directional encoder so key_size is 2*hidden_size
#         key_size = 2 * hidden_size if key_size is None else key_size
#         query_size = hidden_size if query_size is None else query_size
#
#         self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
#         self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
#         self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
#
#         # to store attention scores
#         self.alphas = None
#
#     def forward(self, query=None, proj_key=None, value=None, mask=None):
#         assert mask is not None, "mask is required"
#
#         # We first project the query (the decoder state).
#         # The projected keys (the encoder states) were already pre-computated.(반복 계산을 줄이기 위해, 미리 계산해 둔다)
#         query = self.query_layer(query)
#
#         # Calculate scores.
#         scores = self.energy_layer(torch.tanh(query + proj_key))
#         scores = scores.squeeze(2).unsqueeze(1)
#
#         # Mask out invalid positions.
#         # The mask marks valid positions so we invert it using `mask & 0`.
#         scores.data.masked_fill_(mask == 0, -float('inf'))
#
#         # Turn scores to probabilities.
#         alphas = F.softmax(scores, dim=-1)
#         self.alphas = alphas        
#
#         # The context vector is the weighted sum of the values.
#         context = torch.bmm(alphas, value)
#
#         # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
#         return context, alphas
#
#
# class BahdanauAttention__(nn.Module):
#     # BahdanauAttention
#     # https://sigmoidal.io/implementing-additive-attention-in-pytorch/  --> hccho: batch처리할 수 있도록 수정
#     def __init__(self, encoder_dim, decoder_dim):          
#         super().__init__()          
#
#         self.encoder_dim = encoder_dim          
#         self.decoder_dim = decoder_dim          
#         self.v = torch.nn.Parameter(torch.rand(self.decoder_dim))          
#         self.W_1 = torch.nn.Linear(self.decoder_dim, self.decoder_dim,bias=False)          
#         self.W_2 = torch.nn.Linear(self.encoder_dim, self.decoder_dim,bias=False)          
#
#     def forward(self,query,value):
#         '''
#             query: [N,decoder_dim]  <---- dedocer hidden   
#             values: [N,Te, encoder_dim] <---- encoder output
#         '''
#         # self.W_2(values) --> processed key는 decoder time step에 의존하는 query와 달리 decoder time step에 의존하지 않는다.
#         # 바깥에서 계산해서 넘기는 것도 가능하다. 
#
#         query = query.reshape(-1,1,query.size(-1))  # broadcasting을 위해 reshape   (N,1,decoder_dim]
#         weights = self.W_1(query) + self.W_2(values) # --> [N,Te, decoder_dim]          
#         weights = torch.tanh(weights) @ self.v # [Te, decoder_dim]@ [decoder_dim]  ==>  [Te]
#
#
#         weights = F.softmax(weights, dim=-1) # [N,Te]
#         context = torch.bmm(weights.unsqueeze(dim=1), values).squeeze(dim=1)
#         return context, weights

