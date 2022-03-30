import torch
from torch import nn
from networks.bilstm_network import BiLSTM
from itertools import zip_longest
from networks.crf import CRF
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, out_size,num_layers,device,dropout,use_dropout=True,use_norm = True,use_bert= False,fine_tuning =False,bert_model_dir = ""):
        """初始化参数：
        BiLSTM-CRF其实就是一个CRF模型，只不过用BiLSTM得到状态特征值sk，用反向传播算法更新转移特征值tk 。
            vocab_size:字典的大小
            embedding_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM_CRF, self).__init__()
        self.bilstm = BiLSTM(vocab_size=vocab_size,
                             embedding_size=embedding_size,
                             hidden_size=hidden_size,
                             out_size=out_size,
                             num_layers=num_layers,
                             dropout=dropout,
                             use_dropout=use_dropout,
                             use_norm=use_norm,
                             use_bert=use_bert,
                             fine_tuning = fine_tuning,
                             bert_model_dir=bert_model_dir)
        self.device = device
        self.crf = CRF(out_size,device)
    def forward(self,  batch_sentences, sentence_lengths):
        """
        求出了每一帧对应到每种tag的发射矩阵 也就是每个字对应发射到每一种tag的概率 但是相加和不为1
        LSTM的输出——sentence的每个word经BiLSTM后，对应于每个label的得分
        :param batch_sentences:
        :param sentence_lengths:
        :return:
        """
        #
        #用于后续计算损失函数
        emission = self.bilstm(batch_sentences, sentence_lengths)   # [batch_size, length, out_size]

        # 计算CRF scores, 这个scores大小为[batch_size, length, out_size, out_size]
        # 也就是每个字对应对应一个 [out_size, out_size]的矩阵
        # 这个矩阵第i行第j列的元素的含义是：上一时刻tag为i，这一时刻tag为j的分数
        crf_scores = self.crf(emission)
        return crf_scores
    def test(self, test_batch_sentences, lengths, tag_to_index):
        """
        使用维特比算法进行解码
        CRF层将BiLSTM的Emission_score作为输入，输出符合标注转移约束条件的、最大可能的预测标注序列。
        :param test_batch_sentences:
        :param lengths:
        :param tag_to_id:
        :return:
        """
        """使用维特比算法进行解码"""
        emission = self.bilstm(test_batch_sentences, lengths)  # [batch_size, length, out_size]
        crf_scores = self.crf(emission)
        tag_ids = self.crf.viterbi_decode(emission,lengths,tag_to_index)
        return crf_scores,tag_ids
