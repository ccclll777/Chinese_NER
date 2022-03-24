# coding:utf-8
import codecs
import re
import pandas as pd
import numpy as np
from collections.abc import Iterable
import os
from sklearn.model_selection import train_test_split
import json
base_dir = os.getcwd()
class DataProcessingCLUE():
    def __init__(self):
        self.train_data_file = base_dir+'/data/clue/train.json'
        self.valid_data_file = base_dir + '/data/clue/dev.json'
        self.test_data_file = base_dir + '/data/clue/test.json' #由于测试集没有进行标注 所以我们切分10%的训练集作为测试集
        #训练集
        self.x_train = []
        self.y_train = []
        #验证集
        self.x_valid = []
        self.y_valid = []
        #测试集
        self.x_test = []
        self.y_test = []
        self.tag_to_id = {'o': 0,'B-name': 1, 'B-organization': 2, 'B-address': 3,'B-company': 4,'B-government': 5,
                  'B-book': 6,'B-game': 7,'B-movie': 8, 'B-position': 9,"B-scene":10,
                    'I-name': 11, 'I-organization': 12, 'I-address': 13,'I-company': 14,'I-government': 15,
                  'I-book': 16,'I-game': 17,'I-movie': 18, 'I-position': 19,"I-scene":20}
        self.id_to_tag = {0: 'o',1: 'B-name', 2: 'B-organization',3: 'B-address',4: 'B-company',5: 'B-government',
                  6: 'B-book',7: 'B-game',8: 'B-movie',9: 'B-position',10:"B-scene",
                          11: 'I-name', 12: 'I-organization', 13: 'I-address', 14: 'I-company', 15: 'I-government',
                          16: 'I-book', 17: 'I-game', 18: 'I-movie', 19: 'I-position', 20: "I-scene",
                          }
        self.max_len = 50 #最长句子的长度

    def read_json(self,input_file,mode="train"):
        """
        读取数据集
        :param input_file:  文件路径
        :param mode:  训练集  验证集 or 测试集  train valid test
        :return:
        """
        with open(input_file, 'r') as f:
            for line in f:
                line = json.loads(line.strip())
                text = line['text']
                tags_entities = line.get('label', None) #找到tag
                words = list(text)
                tags = ['o'] * len(words) #现将所有的位置都标记为o 然后再去提取具体的标记
                if tags_entities is not None:
                    for key, value in tags_entities.items(): #遍历所有的tag 和对应的位置  key为实体类别（地点 名字） value为具体的实体文本
                        for sub_name, sub_index in value.items():#sub_name 为标记的实体的文本 sub_index为实体所在位置
                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index:end_index + 1]) == sub_name
                                if start_index == end_index:#如果只有一个文本 则标注为Begin
                                    tags[start_index] = 'B-' + key
                                else: #如果不是结尾 则需要标注中间文本B I O
                                    tags[start_index] = 'B-' + key
                                    tags[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
                """
                训练集 验证集 测试集分别存储
                """
                if mode == "train":
                    self.x_train.append(words)
                    self.y_train.append(tags)
                elif mode == "valid":
                    self.x_valid.append(words)
                    self.y_valid.append(tags)
                elif mode == "test":
                    self.x_test.append(words)
                    self.y_test.append(tags)

    def flatten(self,x):
        """
        统计句子中的所有字
        :param self:
        :param x:
        :return: 返回为list 包含data中所有的字
        """
        result = []
        for el in x:
            if isinstance(x, Iterable) and not isinstance(el, str):
                result.extend(self.flatten(el))
            else:
                result.append(el)
        return result
    def word_to_id(self):
        """
        构建word to index 和index to word的映射 只处理训练集和验证集
        :return:
        """
        all_words = self.flatten(self.x_train+self.x_valid)
        series_all_word = pd.Series(all_words)
        series_all_word = series_all_word.value_counts() #对所有字统计频率
        set_words = series_all_word.index # 去除重复 每个字只保留一次
        set_indexs = range( len(set_words))
        #构建word to index 和index to word的映射
        self.word_to_index = pd.Series(set_indexs, index=set_words)
        self.index_to_word = pd.Series(set_words, index=set_indexs)
    def get_train_valid_test_set(self):
        """
        切分测试集  由于提供的测试集没有标注 所以随机切分10%的训练集作为测试集
        :return:
        """
        x = np.array(self.x_train,dtype=object)
        y = np.array(self.y_train,dtype=object)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.1, random_state=43)
    def save_train_valid_test_set(self):
        """
        使用numpy保存
        :return:
        """
        dir = base_dir +"/data/clue/"

        word_to_index = self.word_to_index.to_dict()
        index_to_word = self.index_to_word.to_dict()
        np.save(dir+'word_to_index.npy', word_to_index)
        np.save(dir + 'index_to_word.npy', index_to_word)
        np.save(dir+'tag_to_id.npy', self.tag_to_id)
        np.save(dir + 'id_to_tag.npy', self.id_to_tag)
        np.save(dir+"x_train.npy", self.x_train)
        np.save(dir + "y_train.npy", self.y_train)
        np.save(dir + "x_test.npy", self.x_test)
        np.save(dir + "y_test.npy", self.y_test)
        self.x_valid =np.array(self.x_valid,dtype=object)
        self.y_valid = np.array(self.y_valid, dtype=object)
        np.save(dir + "x_valid.npy", self.x_valid)
        np.save(dir + "y_valid.npy", self.y_valid)
data_processing = DataProcessingCLUE()
data_processing.read_json(data_processing.train_data_file,mode="train")
data_processing.read_json(data_processing.valid_data_file,mode="valid")
# data_processing.read_json(data_processing.test_data_file,mode="test")#由于测试集没有进行标注 所以我们切分10%的训练集作为测试集
"""构建word to index index to word的列表 测试集不参与构建"""
data_processing.word_to_id()
#由于测试集没有进行标注 所以我们切分10%的训练集作为测试集
data_processing.get_train_valid_test_set()
data_processing.save_train_valid_test_set()
print(len(data_processing.x_valid))
print(len(data_processing.y_valid))