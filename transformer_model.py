import numpy as np
import torch.nn  as nn
import torch

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Implement the positional encoding (PE) function.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x


class multi_attention(nn.Module):
    def __init__(self,words_embedding_dim,heads_num=8,qkv_drop=0.,linear_drop = 0.):
        super(multi_attention, self).__init__()
        self.words_embedding_dim = words_embedding_dim
        self.heads_num = heads_num
        self.heads_dim = words_embedding_dim // heads_num
        self.qkv_linear = nn.Linear(self.words_embedding_dim,self.words_embedding_dim,bias=False)
        self.qkv_drop = nn.Dropout(qkv_drop)
        self.linear_drop = nn.Dropout(linear_drop)
        self.linear = nn.Linear(words_embedding_dim,words_embedding_dim)


    def forward(self,x):

        batch, words_num,embedding_dim = x.shape
        Q = self.qkv_linear(x).reshape(batch,words_num,self.heads_num,self.heads_dim).transpose(1,2)
        K = self.qkv_linear(x).reshape(batch,words_num,self.heads_num,self.heads_dim).transpose(1,2)
        V = self.qkv_linear(x).reshape(batch,words_num,self.heads_num,self.heads_dim).transpose(1,2)

        attn = (Q@K.transpose(-2,-1))*(self.heads_dim** -0.5)
        attn = attn.softmax(dim =-1)
        attn = self.qkv_drop(attn)

        x = (attn@V).transpose(1,2).reshape(batch,words_num,embedding_dim)
        x = self.linear(x)

        x = self.linear_drop(x)
        return x




class mlp(nn.Module):
    def __init__(self,input_dim,output_dim,hiden_dim_ratio = 4,linear_bias=True,dropout = 0.5):
        super(mlp, self).__init__()
        self.input_dim = input_dim
        self.hiden_dim = hiden_dim_ratio*self.input_dim
        self.output_dim = output_dim

        self.bias = linear_bias
        self.linear1 = nn.Linear(self.input_dim, self.hiden_dim, bias=self.bias)
        self.linear2 = nn.Linear(self.hiden_dim, self.output_dim, bias=self.bias)
        self.drop = nn.Dropout(dropout)
        self.gelu = nn.GELU()
    def forward(self,x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x





class attention_block(nn.Module):
    def __init__(self,words_enbedding_size,
                 mlp_input_dim,
                 mlp_output_dim,
                 mlp_hiden_dim_ratio=4,
                 attn_heads_num=8,
                 attn_qkv_drop=0.1,
                 attn_linear_drop = 0.1,
                 mlp_linear_bias=True,
                 mlp_linear_dropout = 0.1):
        super(attention_block,self).__init__()

        self.layer_norm1 = nn.LayerNorm(words_enbedding_size)
        self.layer_norm2 = nn.LayerNorm(words_enbedding_size)

        # self.layer_norm1 = nn.BatchNorm1d(words_enbedding_size)
        # self.layer_norm2 = nn.BatchNorm1d(words_enbedding_size)

        self.attn = multi_attention(words_enbedding_size, heads_num=attn_heads_num, qkv_drop=attn_qkv_drop, linear_drop = attn_linear_drop)
        self.mlp = mlp(mlp_input_dim,mlp_output_dim,mlp_hiden_dim_ratio,linear_bias=mlp_linear_bias,dropout=mlp_linear_dropout)


    def forward(self,x):
        x = x+self.attn(self.layer_norm2(x))

        x = x+self.mlp(self.layer_norm2(x))

        return x




class transformer_model(nn.Module):
      def __init__(self,
                   words_embedding_size,
                   class_num,
                   depth1,
                   mlp_input_dim,
                   mlp_output_dim,
                   attn_heads_num=10,
                   attn_qkv_drop=0.,
                   attn_linear_drop = 0.,
                   mlp_linear_bias=True,
                   mlp_hiden_dim_ratio=4,
                   mlp_linear_dropout = 0.,
                   cla_drop =0.2
                   ):
          super(transformer_model, self).__init__()
          self.embedding_size = words_embedding_size
          self.blocks1 = nn.Sequential(*[
              attention_block(self.embedding_size,mlp_input_dim,mlp_output_dim,mlp_hiden_dim_ratio,
                   attn_heads_num,
                   attn_qkv_drop,
                   attn_linear_drop,
                   mlp_linear_bias,
                   mlp_linear_dropout  )
            for i in range(depth1)
          ])
          # !!!!!  操了，这个格式

          self.layer_norm = nn.LayerNorm(self.embedding_size)
          self.cla_linear = nn.Linear(self.embedding_size,self.embedding_size*4,True)
          self.cla_linear1 = nn.Linear(self.embedding_size*4,class_num,bias=True)
          self.drop = nn.Dropout(cla_drop)
          self.softmax = nn.Softmax(dim=1)
          self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_size))
          self.apply(_init_vit_weights)
          self.sigmoid = nn.Sigmoid()
          self.position = PositionalEncoding(940,16)

      def forward(self,x):
          # print(x.shape)
          x = x.reshape(x.shape[0],-1,self.embedding_size)


          cls_token = self.cls_token.expand(x.shape[0], -1, -1)


          x = torch.cat((cls_token, x), dim=1)

          x = self.position(x)

          x = self.blocks1(x)

          cla = x[:, 0]
          cla = self.cla_linear(cla)
          cla = self.drop(cla)
          cla = self.cla_linear1(cla)
          cla = self.sigmoid(cla)





          return cla


def _init_vit_weights(m):
    """
    ViT weights initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)



if __name__ == "__main__":
    model = transformer_model(words_embedding_size = 553,
                              class_num=8,
                              depth1=2,
                              mlp_input_dim=553,
                              mlp_output_dim=553, )
    x = np.random.randint(0,2,size=3318)
    print(model.forward(x))
