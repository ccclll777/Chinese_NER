import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, out_size):
        """初始化参数：
            双向lstm网络
            vocab_size:字典的大小
            embedding_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size) #Embedding层
        #batch_first： 如果是True，则input为(batch, seq, input_size)。默认值为：False（seq_len, batch, input_size）
        self.bilstm = nn.LSTM(embedding_size, hidden_size,
                              batch_first=True,
                              bidirectional=True)

        self.linear = nn.Linear(2*hidden_size, out_size) #将biilist

    def forward(self, batch_sentences, sentence_lengths):
        """

        :param batch_sentences:  句子
        :param sentence_lengths: 这个batch 每个句子的长度
        :return:
        """
        embedding = self.embedding(batch_sentences)  # [batch_size, length, embedding_size]
        #https://zhuanlan.zhihu.com/p/342685890
        #在 pad 之后再使用 pack_padded_sequence 对数据进行处理 避免无效的计算 pad的位置不会进行计算
        packed = pack_padded_sequence(embedding, sentence_lengths, batch_first=True)
        output, (h_n, c_n) = self.bilstm(packed)
        # rnn_out:[B, L, hidden_size*2]
        output, _ = pad_packed_sequence(output, batch_first=True) #pack_padded_sequence 函数的逆向操作。就是把压紧的序列再填充回来

        scores = self.linear(output)  #用线性层映射到输出 # [batch_size, length, out_size]
        return scores

    def test(self, sents_tensor, lengths, _):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        logits = self.forward(sents_tensor, lengths) # [batch_size, length, out_size]
        _, batch_tag_index = torch.max(logits, dim=2) #找到得分最大的index

        return batch_tag_index
