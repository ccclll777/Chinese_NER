import os
from collections import Counter
from collections.abc import Iterable
import torch
import  numpy as np
class DataSet():
    def __init__(self,data_set="clue",base_dir = os.getcwd()):
        """

        :param data_set:  读取那个数据集  clue or msra
        """
        if data_set == "msra":
            """msra数据集"""
            self.tag_list = ["ns","nr","nt","o"]
            self.x_train_file = base_dir +"/data/msra/x_train.npy"
            self.y_train_file = base_dir + "/data/msra/y_train.npy"
            self.x_valid_file = base_dir + "/data/msra/x_valid.npy"
            self.y_valid_file = base_dir + "/data/msra/y_valid.npy"
            self.x_test_file = base_dir + "/data/msra/x_test.npy"
            self.y_test_file = base_dir + "/data/msra/y_test.npy"
            self.tag_to_index = {'o': 0, 'B_ns': 1, 'B_nr': 2, 'B_nt': 3, 'M_nt': 4, 'M_nr': 5,
                              'M_ns': 6, 'E_nt': 7, 'E_nr': 8, 'E_ns': 9, }
            self.index_to_tag = {0: 'o', 1: 'B_ns', 2: 'B_nr', 3: 'B_nt', 4: 'M_nt', 5: 'M_nr',
                              6: 'M_ns', 7: 'E_nt', 8: 'E_nr', 9: 'E_ns'}
        elif data_set == "clue":
            """
                clue数据集
            """
            self.tag_list = ["address", "book", "company","game", "government", "movie","name", "organization", "position","scene","o"]
            self.x_train_file = base_dir +"/data/clue/x_train.npy"
            self.y_train_file = base_dir + "/data/clue/y_train.npy"
            self.x_valid_file = base_dir + "/data/clue/x_valid.npy"
            self.y_valid_file = base_dir + "/data/clue/y_valid.npy"
            self.x_test_file = base_dir + "/data/clue/x_test.npy"
            self.y_test_file = base_dir + "/data/clue/y_test.npy"
            self.tag_to_index = {'o': 0, 'B-name': 1, 'B-organization': 2, 'B-address': 3, 'B-company': 4,
                              'B-government': 5,
                              'B-book': 6, 'B-game': 7, 'B-movie': 8, 'B-position': 9, "B-scene": 10,
                              'I-name': 11, 'I-organization': 12, 'I-address': 13, 'I-company': 14, 'I-government': 15,
                              'I-book': 16, 'I-game': 17, 'I-movie': 18, 'I-position': 19, "I-scene": 20}
            self.index_to_tag = {0: 'o', 1: 'B-name', 2: 'B-organization', 3: 'B-address', 4: 'B-company',
                              5: 'B-government',
                              6: 'B-book', 7: 'B-game', 8: 'B-movie', 9: 'B-position', 10: "B-scene",
                              11: 'I-name', 12: 'I-organization', 13: 'I-address', 14: 'I-company', 15: 'I-government',
                              16: 'I-book', 17: 'I-game', 18: 'I-movie', 19: 'I-position', 20: "I-scene",
                              }
        self.load_data()
    def load_data(self):
        self.x_train = np.load(self.x_train_file,allow_pickle=True).tolist()
        self.y_train = np.load(self.y_train_file,allow_pickle=True).tolist()
        self.x_valid = np.load(self.x_valid_file,allow_pickle=True).tolist()
        self.y_valid = np.load(self.y_valid_file,allow_pickle=True).tolist()
        self.x_test = np.load(self.x_test_file,allow_pickle=True).tolist()
        self.y_test = np.load(self.y_test_file,allow_pickle=True).tolist()
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
    def word_to_id(self,min_freq):
        """
        构建word to index 和index to word的映射 只处理训练集和验证集
        :return:
        """
        all_words = self.flatten(self.x_train + self.x_valid+self.x_test)
        word_counter = Counter()
        word_counter.update(all_words)
        # series_all_word = pd.Series(all_words)
        # series_all_word = series_all_word.value_counts() #对所有字统计频率
        # max_size = min(max_freq, len(word_counter)) if max_freq else None
        # words = word_counter.most_common(max_size)
        self.word_to_index = dict()
        self.index_to_word = dict()
        if min_freq >0:
            for key, cnts in list(word_counter.items()):  # list is important here
                if cnts < min_freq:
                    del word_counter[key]
        offset = len(self.word_to_index)
        ls = word_counter.most_common(int(10000))
        self.word_to_index.update({w: i + offset for i, (w, _) in enumerate(ls)})
        self.index_to_word.update({i + offset: w for i, (w, _) in enumerate(ls)})
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
        self.word_to_index['<CLS>'] = len(self.word_to_index)
        self.index_to_word[len(self.word_to_index)-1] = '<CLS>'
        self.word_to_index['<SEP>'] = len(self.word_to_index)
        self.index_to_word[len(self.word_to_index)-1] = '<SEP>'
        self.tag_to_index['<CLS>'] = len(self.tag_to_index)
        self.index_to_tag[len(self.tag_to_index) - 1] = '<CLS>'
        self.tag_to_index['<SEP>'] = len(self.tag_to_index)
        self.index_to_tag[len(self.tag_to_index) - 1] = '<SEP>'
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
            word_lists[i].append("<SEP>")
            if not test:  # 如果是测试数据，就不需要加end token了
                tag_lists[i].append("<SEP>")
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
    def transformer_word_to_index(self,list,maps):
        """
         transformer训练时，将句子或者tags映射成index 并且构造mask
        :param words_list:
        :param maps:  映射的dict word_to_index 或者tag_to_index
        :return:
        """
        PAD = maps.get('<PAD>')
        UNK = maps.get('<UNK>')
        max_len = len(list[0])
        batch_size = len(list)
        batch_tensor = torch.ones(batch_size, max_len).long() * PAD
        """
        构造mask
        """
        mask = torch.LongTensor(batch_size, max_len).fill_(0)
        for i, line in enumerate(list):
            for j, word in enumerate(line):
                batch_tensor[i][j] = maps.get(word, UNK)
            mask[i, :len(line)] = torch.tensor([1] * len(line), dtype=torch.long)
        # words_list各个元素的长度
        lengths = [len(line) for line in list]
        return batch_tensor, lengths,mask

    def bert_crf_word_to_index(self,list,maps):
        """
         bert训练时，将句子或者tags映射成index 并且构造mask 然后还需要构造token_type_ids 由于我们只有一个句子   所以全0即可
        :param words_list:
        :param maps:  映射的dict word_to_index 或者tag_to_index
        :return:
        """
        PAD = maps.get('<PAD>')
        UNK = maps.get('<UNK>')
        max_len = len(list[0])
        batch_size = len(list)
        batch_tensor = torch.ones(batch_size, max_len).long() * PAD
        """
        构造mask
        """
        mask = torch.LongTensor(batch_size, max_len).fill_(0)
        token_type_ids = torch.LongTensor(batch_size, max_len).fill_(0)
        for i, line in enumerate(list):
            for j, word in enumerate(line):
                batch_tensor[i][j] = maps.get(word, UNK)
            mask[i, :len(line)] = torch.tensor([1] * len(line), dtype=torch.long)
        # words_list各个元素的长度
        lengths = [len(line) for line in list]
        return batch_tensor, lengths,mask,token_type_ids

    def word_to_token(self,sentence,deep_model = False):
        """
        将句子转化为token，用于单个句子生成命名实体识别的结果。
        :param sentence:
        :param deep_model:  用的是否是深度学习的模型
        :return:
        """
        index_list = []
        START = self.word_to_index.get('<CLS>')
        END = self.word_to_index.get('<SEP>')
        PAD = self.word_to_index.get('<PAD>')
        UNK = self.word_to_index.get('<UNK>')
        if deep_model:
            #如果不是HMM和CRF 则需要添加start和end标记
            index_list.append(START)
        for word in sentence:
            index_list.append(self.word_to_index.get(word, UNK))
        if deep_model:
            #如果不是HMM和CRF 则需要添加start和end标记
            index_list.append(END)
        return index_list

# data_set = DataSet()
# data_set.load_data()