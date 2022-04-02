import torch
import torch.nn as nn
from networks.bert_base_network import BaseBertModel
from networks.crf import CRF


class BertCRF(BaseBertModel):
    def __init__(self,
                 hidden_size,
                 num_tags,
                 # num_layers,
                 device,
                 dropout,
                 bert_model_dir = ""):
        super(BertCRF, self).__init__(bert_dir=bert_model_dir, dropout=dropout)
        self.device = device
        bert_out_dims = self.bert_config.hidden_size
        self.mid_linear = nn.Sequential(
                nn.Linear(bert_out_dims, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout))
        self.layer = nn.Linear(hidden_size,num_tags) #映射到标签
        init_blocks = [self.mid_linear, self.layer]
        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)
        #CRF层
        self.crf = CRF(out_size=num_tags,device = self.device)
    def forward(self,
                input_index, input_mask,
                token_type_ids):
        bert_outputs = self.bert_module(
            input_ids=input_index,
            attention_mask=input_mask,
            token_type_ids=token_type_ids)
        output = bert_outputs[0]  # [batchsize, max_len, 768]
        output = self.mid_linear(output)  # [batch_size, max_len, hidden_size]
        emission = self.layer(output)  #
        crf_scores = self.crf(emission)
        return crf_scores
    def test(self, input_index, input_mask,token_type_ids,lengths, tag_to_index):
        """
        使用维特比算法进行解码
        CRF层将Bert的Emission_score作为输入，输出符合标注转移约束条件的、最大可能的预测标注序列。

        :return:
        """
        """使用维特比算法进行解码"""
        emission = self.forward(input_index, input_mask,token_type_ids)  #batch_size*len*128 # [batch_size, length, out_size]
        crf_scores = self.crf(emission)
        tag_ids = self.crf.viterbi_decode(emission,lengths,tag_to_index)
        return crf_scores,tag_ids