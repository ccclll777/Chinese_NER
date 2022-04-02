import os
import torch.nn as nn
from transformers import BertModel
class BaseBertModel(nn.Module):
    def __init__(self, bert_dir, dropout):
        super(BaseBertModel, self).__init__()
        # bert_config_path = os.path.join(bert_dir, 'config.json')
        #加载与训练的Bert
        self.bert_module = BertModel.from_pretrained(bert_dir, output_hidden_states=True,
                                                     hidden_dropout_prob=dropout)
        self.bert_config = self.bert_module.config

    @staticmethod
    def _init_weights(blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)