import torch.nn as nn
from networks.crf import CRF
from networks.transformer_modules import TokenEmbedding,TransformerEncoder,PositionalEncoding

class Transformer_CRF(nn.Module):

    def __init__(self, vocab_size, out_size, num_blocks, d_model, num_heads, feedforward_dim,device, use_dropout=True,dropout=0.1):
        super(Transformer_CRF, self).__init__()
        # self.model_dim = d_model
        self.device = device
        self.use_dropout = use_dropout
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        if self.use_dropout :
            self.drop = nn.Dropout(dropout)
        self.encoder = TransformerEncoder(num_blocks, d_model, num_heads, feedforward_dim, dropout)
        self.linear = nn.Linear(d_model, out_size)
        self.crf = CRF(out_size=out_size,device = self.device)

    def build_features(self, input_index, input_mask):
        embedding = self.embedding(input_index)
        embedding = self.pos_encoding(embedding)
        if self.use_dropout:
            embedding = self.drop(embedding)
        embedding = embedding * input_mask.float().unsqueeze(2)
        attention_mask = get_attention_pad_mask(input_index, input_index)
        emission = self.encoder(embedding, attention_mask)
        emission= self.linear(emission)
        return emission

    def forward(self, input_index, input_mask):
        emission = self.build_features(input_index, input_mask)#batch_size*len*128

        crf_scores = self.crf(emission)
        return crf_scores

    def test(self, input_index, input_mask,lengths, tag_to_index):
        """
                使用维特比算法进行解码
        CRF层将Transformer的Emission_score作为输入，输出符合标注转移约束条件的、最大可能的预测标注序列。
        :param input_index:  word_list
        :param input_mask:  word mask
        :param lengths:  每个句子的长度
        :param tag_to_index: tag到index的映射
         :return:
        """
        """使用维特比算法进行解码"""
        emission = self.build_features(input_index, input_mask)  #batch_size*len*128 # [batch_size, length, out_size]
        crf_scores = self.crf(emission)
        tag_ids = self.crf.viterbi_decode(emission,lengths,tag_to_index)
        return crf_scores,tag_ids
def get_attention_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k