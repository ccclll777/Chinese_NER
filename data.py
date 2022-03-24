
import os
import pickle
import torch
import  numpy as np
base_dir = os.getcwd()
class DataSet():
    def __init__(self,data_set="clue"):
        """

        :param data_set:  读取那个数据集  clue or msra
        """
        if data_set == "msra":
            """msra数据集"""
            self.x_train_file = base_dir +"/data/msra/x_train.npy"
            self.y_train_file = base_dir + "/data/msra/y_train.npy"
            self.x_valid_file = base_dir + "/data/msra/x_valid.npy"
            self.y_valid_file = base_dir + "/data/msra/y_valid.npy"
            self.x_test_file = base_dir + "/data/msra/x_test.npy"
            self.y_test_file = base_dir + "/data/msra/y_test.npy"
            self.index_to_tag_file = base_dir + "/data/msra/id_to_tag.npy"
            self.tag_to_index_file = base_dir + "/data/msra/tag_to_id.npy"
            self.index_to_word_file = base_dir + "/data/msra/index_to_word.npy"
            self.word_to_index_file = base_dir + "/data/msra/word_to_index.npy"
        elif data_set == "clue":
            """
                clue数据集
            """
            self.x_train_file = base_dir +"/data/clue/x_train.npy"
            self.y_train_file = base_dir + "/data/clue/y_train.npy"
            self.x_valid_file = base_dir + "/data/clue/x_valid.npy"
            self.y_valid_file = base_dir + "/data/clue/y_valid.npy"
            self.x_test_file = base_dir + "/data/clue/x_test.npy"
            self.y_test_file = base_dir + "/data/clue/y_test.npy"
            self.index_to_tag_file = base_dir + "/data/clue/id_to_tag.npy"
            self.tag_to_index_file = base_dir + "/data/clue/tag_to_id.npy"
            self.index_to_word_file = base_dir + "/data/clue/index_to_word.npy"
            self.word_to_index_file = base_dir + "/data/clue/word_to_index.npy"
        self.load_data()
    def load_data(self):
        self.x_train = np.load(self.x_train_file,allow_pickle=True).tolist()
        self.y_train = np.load(self.y_train_file,allow_pickle=True).tolist()
        self.x_valid = np.load(self.x_valid_file,allow_pickle=True).tolist()
        self.y_valid = np.load(self.y_valid_file,allow_pickle=True).tolist()
        self.x_test = np.load(self.x_test_file,allow_pickle=True).tolist()
        self.y_test = np.load(self.y_test_file,allow_pickle=True).tolist()
        self.index_to_tag = np.load(file =self.index_to_tag_file, allow_pickle=True).item()
        self.tag_to_index = np.load(file =self.tag_to_index_file, allow_pickle=True).item()
        self.index_to_word = np.load(file =self.index_to_word_file, allow_pickle=True).item()
        self.word_to_index = np.load(file =self.word_to_index_file, allow_pickle=True).item()
    def word_to_features(self,sentence, i):
        """
        抽取单个字的特征 crf使用
        :param sentence:
        :param i:
        :return:
        """
        word = sentence[i]
        prev_word = "<s>" if i == 0 else sentence[i - 1]
        next_word = "</s>" if i == (len(sentence) - 1) else sentence[i + 1]
        # 使用的特征：
        # 前一个词，当前词，后一个词，
        # 前一个词+当前词， 当前词+后一个词
        features = {
            'w': word,
            'w-1': prev_word,
            'w+1': next_word,
            'w-1:w': prev_word + word,
            'w:w+1': word + next_word,
            'bias': 1
        }
        return features
    def decode_sentence(self,sentence):
        """
        对句子进行解码
        :param sentence:
        :return:
        """
        return [self.index_to_word[i] for i in sentence]

    def decode_sentence_list(self,sentence_list):
        """
        将sentence句子解码变成文本
        :return:
        """
        # list =
        return [self.decode_sentence(i) for i in sentence_list]
    def decode_tag(self,tag):
        """
        解码 tag
        :param tag:
        :return:
        """
        return [self.index_to_tag[i] for i in tag]
    def decode_tag_list(self,tag_list):
        return [self.decode_tag(i) for i in tag_list]
    def extend_maps(self):
        """
        BILSTM模型训练的时候需要在word_to_index和tag_to_index加入PAD和UNK
        如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
        :param for_crf:
        :return:
        """
        self.word_to_index['<UNK>'] = len(self.word_to_index)

        self.index_to_word[len(self.word_to_index)-1] = '<UNK>'

        self.word_to_index['<PAD>'] = len(self.word_to_index)
        self.index_to_word[len(self.word_to_index) - 1] = '<PAD>'

        self.tag_to_index['<UNK>'] = len(self.tag_to_index)
        self.index_to_tag[len(self.tag_to_index) - 1] = '<UNK>'

        self.tag_to_index['<PAD>'] = len(self.tag_to_index)
        self.index_to_tag[len(self.tag_to_index) - 1] = '<PAD>'
        # 如果是加了CRF的bilstm  那么还要加入<start> 和 <end>token
        self.word_to_index['<START>'] = len(self.word_to_index)
        self.index_to_word[len(self.word_to_index)-1] = '<START>'
        self.word_to_index['<END>'] = len(self.word_to_index)
        self.index_to_word[len(self.word_to_index)-1] = '<END>'
        self.tag_to_index['<START>'] = len(self.tag_to_index)
        self.index_to_tag[len(self.tag_to_index) - 1] = '<START>'
        self.tag_to_index['<END>'] = len(self.tag_to_index)
        self.index_to_tag[len(self.tag_to_index) - 1] = '<END>'
    def prepocess_data_for_lstm_crf(self,word_lists, tag_lists, test=False):
        """
        给数据添加结束符号
        如果是测试数据，就不需要加end token了
        :param word_lists:
        :param tag_lists:
        :param test:
        :return:
        """
        for i in range(len(word_lists)):
            word_lists[i].append("<END>")
            if not test:  # 如果是测试数据，就不需要加end token了
                tag_lists[i].append("<END>")
        return word_lists, tag_lists
    def bilstm_crf_word_to_index(self,list,maps):
        """
         bilstm_crf训练时，将句子或者tags映射成index
        :param words_list:
        :param maps:  映射的dict word_to_index 或者tag_to_index
        :return:
        """
        PAD = maps.get('<PAD>')
        UNK = maps.get('<UNK>')
        max_len = len(list[0])
        batch_size = len(list)
        batch_tensor = torch.ones(batch_size, max_len).long() * PAD
        for i, line in enumerate(list):
            for j, word in enumerate(line):
                batch_tensor[i][j] = maps.get(word, UNK)
        # words_list各个元素的长度
        lengths = [len(line) for line in list]
        return batch_tensor, lengths
# data_set = DataSet()
# data_set.load_data()