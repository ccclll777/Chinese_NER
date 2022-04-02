import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from pytorch_pretrained_bert import BertModel
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, out_size,num_layers,dropout =0.1,use_dropout=True,use_norm = True,use_bert = False,fine_tuning=False,bert_model_dir = ""):
        """初始化参数：
            双向lstm网络 +dropout +归一化
            vocab_size:字典的大小
            embedding_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM, self).__init__()
        """
        是否使用bert作为词嵌入
        """
        self.use_dropout = use_dropout
        self.use_bert = use_bert
        self.use_norm = use_norm
        self.fine_tuning = fine_tuning
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        if use_bert:
            self.bert_embedding = BertModel.from_pretrained(bert_model_dir)
            self.embedding_size = self.bert_embedding.config.hidden_size
            for param in self.bert_embedding.parameters():
                param.requires_grad = self.fine_tuning
            # self.embedding_size = self.bert_embedding.
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size) #Embedding层
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)
            # batch_first： 如果是True，则input为(batch, seq, input_size)。默认值为：False（seq_len, batch, input_size）
            self.bilstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size,
                                  batch_first=True,
                                  bidirectional=True,
                                  num_layers=num_layers,
                                  dropout=dropout
                                  )
        else:
            self.bilstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size,
                                  batch_first=True,
                                  bidirectional=True,
                                  num_layers=num_layers )


        if self.use_norm:
            self.layer_norm = LayerNorm(self.hidden_size * 2)
        self.linear = nn.Linear(2*self.hidden_size, out_size) #将biilist
    def bert_encoder(self, x):
        """
        使用预训练的bert进行encoder
        :param x: [batch）size, sent_len]
        :return: [batch_size, sent_len, bert-base-chinese]
        """
        with torch.no_grad():
            encoded_layer, _  = self.bert_embedding(x)
            encoded = encoded_layer[-1]
        return encoded
    def forward(self, batch_sentences, sentence_lengths):
        """

        :param batch_sentences:  句子
        :param sentence_lengths: 这个batch 每个句子的长度
        :return:
        """
        if self.use_bert:
            embedding = self.bert_encoder(batch_sentences) #[batch_size, length, 786]
        else:
            embedding = self.embedding(batch_sentences)  # [batch_size, length, embedding_size]
        if self.use_dropout:
            embedding = self.dropout(embedding)
        #https://zhuanlan.zhihu.com/p/342685890
        #在 pad 之后再使用 pack_padded_sequence 对数据进行处理 避免无效的计算 pad的位置不会进行计算
        packed = pack_padded_sequence(embedding, sentence_lengths, batch_first=True)
        output, (h_n, c_n) = self.bilstm(packed)

        # rnn_out:[B, L, hidden_size*2]
        output, _ = pad_packed_sequence(output, batch_first=True) #pack_padded_sequence 函数的逆向操作。就是把压紧的序列再填充回来
        if self.use_norm:
            output = self.layer_norm(output)
        scores = self.linear(output)  #用线性层映射到输出 # [batch_size, length, out_size]
        return scores

    # def test(self, sents_tensor, lengths, _):
    #     """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
    #     logits = self.forward(sents_tensor, lengths) # [batch_size, length, out_size]
    #     _, batch_tag_index = torch.max(logits, dim=2) #找到得分最大的index
    #
    #     return batch_tag_index
