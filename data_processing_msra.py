# coding:utf-8
import codecs
import re
import pandas as pd
import numpy as np
from collections.abc import Iterable
import os
from collections import Counter
from sklearn.model_selection import train_test_split
base_dir = os.getcwd()
class DataProcessingMSRA():
    def __init__(self):
        self.train_data_file = base_dir+'/data/msra/train.txt'
        self.train_cleaned_data_file = base_dir+'/data/msra/word_tag.txt'
        self.datas = []
        self.labels = []
        self.line_data = []
        self.line_label = []

        self.max_len = 100 #最长句子的长度
    def msra_word_tag(self):
        """
        处理msra数据集，由于他的分词不是字粒度的，所以需要进行切分,然后变成BIO标记法 (字/label )
        :return:
        """
        input_data = open(file = self.train_data_file, mode='r', encoding='utf-8')
        output_data = open(file = self.train_cleaned_data_file,mode= 'w', encoding='utf-8')
        for line in input_data.readlines():
            # line=re.split('[，。；！：？、‘’“”]/[o]'.decode('utf-8'),line.strip())
            line = line.strip().split()
            if len(line) == 0:
                continue
            for word in line:
                word = word.split('/')
                if word[1] != 'o': #o为除了人名 地名 组织名的其他
                    if len(word[0]) == 1: #如果只有一个字符 说明只可能是Begin+ns/nr/nt
                        output_data.write(word[0] + "/B_" + word[1] + " ")
                    elif len(word[0]) == 2:#如果有两个个字符 说明是Begin+ns/nr/nt   End+ns/nr/nt
                        output_data.write(word[0][0] + "/B_" + word[1] + " ")
                        output_data.write(word[0][1] + "/E_" + word[1] + " ")
                    else:#如果有多于两个自负  说明是Begin+ns/nr/nt  M+ns/nr/nt   End+ns/nr/nt
                        output_data.write(word[0][0] + "/B_" + word[1] + " ")
                        for j in word[0][1:len(word[0]) - 1]:
                            output_data.write(j + "/M_" + word[1] + " ")
                        output_data.write(word[0][-1] + "/E_" + word[1] + " ")
                else:  #是其他字符 将每个字切分 然后标记为/o
                    for j in word[0]:
                        output_data.write(j + "/o" + " ")
            output_data.write('\n')
        input_data.close()
        output_data.close()
    def clean_word_tag(self):
        """
        将数据集中的每句话，切分为单字和tag的组合 便于训练时使用
        :return:
        """
        input_data = codecs.open(self.train_cleaned_data_file, mode='r', encoding='utf-8')
        for line in input_data.readlines():
            #将一整句话用标点符合切分成 单个句子
            line = re.split('[，。；！：？、‘’“”]/[o]', line.strip())
            #判断句子中有没有有意义的标记 nr ns nt  如果没有的话 则不把这句话作为训练集
            for word_label_pair in line:
                word_label_pair = word_label_pair.strip().split()
                if len(word_label_pair) == 0:
                    continue
                data = []
                label = []
                num_not_o = 0 #不是o的数量
                for word in word_label_pair:
                    word = word.split('/')
                    data.append(word[0])
                    label.append(word[1])
                    if word[1] != 'o':
                        num_not_o += 1
                if num_not_o != 0 and len(data)< self.max_len:
                    # 如果数据集只有o 是不会存入训练集的 如果数据长度大于max len 也不会存入
                    self.datas.append(data)
                    self.labels.append(label)
    # def flatten(self,x):
    #     """
    #     统计句子中的所有字
    #     :param self:
    #     :param x:
    #     :return: 返回为list 包含data中所有的字
    #     """
    #     result = []
    #     for el in x:
    #         if isinstance(x, Iterable) and not isinstance(el, str):
    #             result.extend(self.flatten(el))
    #         else:
    #             result.append(el)
    #     return result
    # def word_to_id(self):
    #     """
    #     构建word to index 和index to word的映射
    #     :return:
    #     """
    #     all_words = self.flatten(self.datas)
    #     word_counter = Counter()
    #     word_counter.update(all_words)
    #
    #     self.word_to_index = dict()
    #     self.index_to_word = dict()
    #     if self.min_freq >0:
    #         for key, cnts in list(word_counter.items()):  # list is important here
    #             if cnts < self.min_freq:
    #                 del word_counter[key]
    #     offset = len(self.word_to_index)
    #     ls = word_counter.most_common(int(10000))
    #     self.word_to_index.update({w: i + offset for i, (w, _) in enumerate(ls)})
    #     self.index_to_word.update({i + offset: w for i, (w, _) in enumerate(ls)})
    def get_train_valid_test_set(self):
        """
        切分训练集 验证集 测试集
        :return:
        """
        #去掉太长的句子
        x = np.array(self.datas,dtype=object)
        y = np.array(self.labels,dtype=object)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.1, random_state=43)
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.x_train, self.y_train, test_size=0.2, random_state=43)
    def save_train_valid_test_set(self):
        """
        使用numpy保存
        :return:
        """
        dir = base_dir +"/data/msra/"
        # np.save(dir+'word_to_index.npy', self.word_to_index)
        # np.save(dir + 'index_to_word.npy', self.index_to_word)
        # np.save(dir+'tag_to_id.npy', self.tag_to_id)
        # np.save(dir + 'id_to_tag.npy', self.id_to_tag)
        np.save(dir+"x_train.npy", self.x_train)
        np.save(dir + "y_train.npy", self.y_train)
        np.save(dir + "x_test.npy", self.x_test)
        np.save(dir + "y_test.npy", self.y_test)
        np.save(dir + "x_valid.npy", self.x_valid)
        np.save(dir + "y_valid.npy", self.y_valid)
data_processing = DataProcessingMSRA()
data_processing.msra_word_tag()
data_processing.clean_word_tag()
# data_processing.word_to_id()
data_processing.get_train_valid_test_set()
data_processing.save_train_valid_test_set()
