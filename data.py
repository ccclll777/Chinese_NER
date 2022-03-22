
import os
import pickle
import  numpy as np
base_dir = os.getcwd()
class DataSet():
    def __init__(self):
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
        self.load_data()
    def load_data(self):
        self.x_train = np.load(self.x_train_file,allow_pickle=True)
        self.y_train = np.load(self.y_train_file,allow_pickle=True)
        self.x_valid = np.load(self.x_valid_file,allow_pickle=True)
        self.y_valid = np.load(self.y_valid_file,allow_pickle=True)
        self.x_test = np.load(self.x_test_file,allow_pickle=True)
        self.y_test = np.load(self.y_test_file,allow_pickle=True)
        self.index_to_tag = np.load(file =self.index_to_tag_file, allow_pickle=True).item()
        self.tag_to_index = np.load(file =self.tag_to_index_file, allow_pickle=True).item()
        self.index_to_word = np.load(file =self.index_to_word_file, allow_pickle=True).item()
        self.word_to_index = np.load(file =self.word_to_index_file, allow_pickle=True).item()

    def extend_maps(self, for_crf=True):
        """
        LSTM模型训练的时候需要在word_to_index和tag_to_index加入PAD和UNK
        如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
        :param for_crf:
        :return:
        """
        self.word_to_index['<UNK>'] = len(self.word_to_index)
        self.word_to_index['<PAD>'] = len(self.word_to_index)
        self.tag_to_index['<UNK>'] = len(self.tag_to_index)
        self.tag_to_index['<PAD>'] = len(self.tag_to_index)
        # 如果是加了CRF的bilstm  那么还要加入<start> 和 <end>token
        if for_crf:
            self.word_to_index['<START>'] = len(self.word_to_index)
            self.word_to_index['<END>'] = len(self.word_to_index)
            self.tag_to_index['<START>'] = len(self.tag_to_index)
            self.tag_to_index['<END>'] = len(self.tag_to_index)

    def word_to_features(self,sentence, i):
        """
        抽取单个字的特征
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

# data_set = DataSet()
# data_set.load_data()