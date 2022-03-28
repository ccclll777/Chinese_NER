import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiheadAttention(nn.Module):
    """ 多头注意力机制 """
    def __init__(self, input_dim, d_model, num_heads):
        """ Construction

        :param input_dim: input dimension
        :param d_model: model dimension
        :param num_heads: number of attention heads
        """
        super(MultiheadAttention, self).__init__()
        assert d_model % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.model_dim = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Query Key Value 仿射矩阵
        self.qkv_proj = nn.Linear(input_dim, 3 * d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        # xavier 初始化权重参数
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        """ 多头注意力计算

        :param x: [batch_size, seq_len, d_model]
        :param mask: [batch_size, seq_len, seq_len]
        :param return_attention: default false
        :return: [batch_size, seq_len, d_model], attention output
        """
        batch_size, seq_len, model_dim = x.size()
        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [batch_size, n_heads, seq_len, d_head]
        q, k, v = qkv.chunk(3, dim=-1)

        # 使用 mask 遮蔽补零位置处的注意力得分
        mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [batch_size, seq_len, n_heads, d_head]
        values = values.reshape(batch_size, seq_len, model_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class EncoderBlock(nn.Module):
    """ Transformer Encoder Block """
    def __init__(self, input_dim, num_heads, feedforward_dim, drop_p=0.0):
        """ Contruction

        :param input_dim: input dimension
        :param num_heads: number of attention heads
        :param feedforward_dim: feed-forward dimension
        :param drop_p: dropout probability
        """
        super(EncoderBlock, self).__init__()

        # multi-head attention sublayer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # position-wise feed-forward sublayer
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, feedforward_dim),
            nn.Dropout(drop_p),
            nn.ReLU(inplace=True),
            nn.Linear(feedforward_dim, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, mask=None):
        # multi-head attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # position-wise feed-forward part
        ff_out = self.feedforward(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder """
    def __init__(self, num_blocks, input_dim, num_heads, feedforward_dim, drop_p=0.0):
        """ Construction

        :param num_blocks: number of encoder block
        :param input_dim: input dimension (model dimension)
        :param num_heads: number of attention heads
        :param feedforward_dim: feed-forward dimension
        :param drop_p: dropout probability
        """
        super(TransformerEncoder, self).__init__()
        self.blocks = nn.ModuleList(
            [EncoderBlock(input_dim, num_heads, feedforward_dim, drop_p) for _ in range(num_blocks)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        """ 输入数据对应的注意力激活特征

        :param x: [batch_size, seq_len, d_model]
        :param mask: [batch_size, seq_len, seq_len]
        :return: attention maps
        """
        attention_maps = []
        for block in self.blocks:
            _, attn_map = block.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = block(x)
        return attention_maps


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model):
        """ Construction

        :param vocab_size: vocab size
        :param d_model: model (embedding) dimension
        """
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.model_dim = d_model

    def forward(self, x):
        # 和 Transformer 论文中保持一致
        return self.embedding(x) * math.sqrt(self.model_dim)
        # return self.embedding(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """ Construction

        :param model_dim: model (embedding) dimension
        :param max_len: maximum length
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer 表示参数属于模型的一部分但不是 Parameter 类型
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention