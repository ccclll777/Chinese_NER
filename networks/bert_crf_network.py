import torch
import torch.nn as nn
from networks.bert_base_network import BaseBertModel
from TorchCRF import CRF


class BertCRFModel(BaseBertModel):
    def __init__(self,
                 hidden_size,
                 num_tags,
                 num_layers,
                 device,
                 dropout,
                 bert_model_dir = ""):
        super(BertCRFModel, self).__init__(bert_dir=bert_model_dir, dropout=dropout)
        self.device = device
        bert_out_dims = self.bert_config.hidden_size
        self.mid_linear = nn.Sequential(
                nn.Linear(bert_out_dims, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout))
        out_dims = hidden_size
        self.layer = nn.Linear(out_dims,num_tags) #映射到标签
        self.criterion = nn.CrossEntropyLoss()
        init_blocks = [self.mid_linear, self.classifier]
        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)
    def forward(self,
                input_index, input_mask,
                token_type_ids):
        bert_outputs = self.bert_module(
            input_ids=input_index,
            attention_mask=input_mask,
            token_type_ids=token_type_ids)
        # 常规
        output = bert_outputs[0]  # [batchsize, max_len, 768]
        output = self.mid_linear(output)  # [batch_size, max_len, hidden_size]
        crf_scores = self.layer(output)  #
        return crf_scores