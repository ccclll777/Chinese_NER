import torch


class HMM(object):
    def __init__(self, tag_size, vocab_size,word_to_index,tag_to_index):
        """Args:
            tag_size: 状态数，这里对应存在的标注的种类
            vocab_size: 观测数，这里对应有多少不同的字
        """
        self.tag_size = tag_size
        self.vocab_size = vocab_size
        self.word_to_index = word_to_index
        self.tag_to_index = tag_to_index
        # 状态转移概率矩阵 A[i][j]表示从i状态转移到j状态的概率
        self.A = torch.zeros(tag_size, tag_size)
        # 观测概率矩阵, B[i][j]表示i状态下生成j观测的概率
        self.B = torch.zeros(tag_size, vocab_size)
        # 初始状态概率  Pi[i]表示初始时刻为状态i的概率
        self.Pi = torch.zeros(tag_size)

    def train(self, word_lists, tag_lists):
        """HMM的训练，即根据训练语料对模型参数进行估计,
           因为我们有观测序列以及其对应的状态序列，所以我们
           可以使用极大似然估计的方法来估计隐马尔可夫模型的参数
        参数:
            word_lists: 列表，其中每个元素由字组成的列表，如 ['担','任','科','员']
            tag_lists: 列表，其中每个元素是由对应的标注组成的列表，如 ['O','O','B-TITLE', 'E-TITLE']
        """

        assert len(tag_lists) == len(word_lists)

        # 估计转移概率矩阵
        for tag_list in tag_lists:
            seq_len = len(tag_list)
            for i in range(seq_len - 1):
                current_tagid = self.tag_to_index[tag_list[i]]
                next_tagid = self.tag_to_index[tag_list[i+1]]
                self.A[current_tagid][next_tagid] += 1
        # 问题：如果某元素没有出现过，该位置为0，这在后续的计算中是不允许的
        # 解决方法：我们将等于0的概率加上很小的数
        self.A[self.A == 0.] = 1e-10
        self.A = self.A / self.A.sum(dim=1, keepdim=True)

        # 估计观测概率矩阵
        for tag_list, word_list in zip(tag_lists, word_lists):
            assert len(tag_list) == len(word_list)
            for tag, word in zip(tag_list, word_list):
                tag_id = self.tag_to_index[tag]
                word_id = self.word_to_index[word]
                self.B[tag_id][word_id] += 1
        self.B[self.B == 0.] = 1e-10
        self.B = self.B / self.B.sum(dim=1, keepdim=True)

        # 估计初始状态概率
        for tag_list in tag_lists:
            init_tagid = self.tag_to_index[tag_list[0]]
            self.Pi[init_tagid] += 1
        self.Pi[self.Pi == 0.] = 1e-10
        self.Pi = self.Pi / self.Pi.sum()

    def test(self, word_lists):
        """
        测试集测试HMM模型
        :param word_lists:
        :return:
        """
        pred_tag_lists = []
        for word_list in word_lists:
            pred_tag_list = self.decoding(word_list, self.word_to_index, self.tag_to_index)
            pred_tag_lists.append(pred_tag_list)
        return pred_tag_lists
    def sentence_ner(self,sentence):
        """
        输入句子进行命名实体识别
        :param sentence:
        :return:
        """
        tag_list = self.decoding(sentence, self.word_to_index, self.tag_to_index)
        return tag_list
    def decoding(self, word_list, word_to_index, tag_to_index):
        """
        使用维特比算法对给定观测序列求状态序列， 这里就是对字组成的序列,求其对应的标注。
        维特比算法实际是用动态规划解隐马尔可夫模型预测问题，即用动态规划求概率最大路径（最优路径）
        这时一条路径对应着一个状态序列
        """
        # 问题:整条链很长的情况下，十分多的小概率相乘，最后可能造成下溢
        # 解决办法：采用对数概率，这样源空间中的很小概率，就被映射到对数空间的大的负数
        #  同时相乘操作也变成简单的相加操作
        A = torch.log(self.A)
        B = torch.log(self.B)
        Pi = torch.log(self.Pi)

        # 初始化 维比特矩阵viterbi 它的维度为[状态数, 序列长度]
        # 其中viterbi[i, j]表示第j个字对应的所有tag（$tag_{1}、tag_{2}...tag_{n}$）出现概率的最大值(每个字对应的预测标签的最大概率)
        seq_len = len(word_list)
        viterbi = torch.zeros(self.tag_size, seq_len)
        # backpointer是跟viterbi一样大小的矩阵
        # backpointer[i, j] 第i行j列表示tag序列的第j个tag对应的字i时，第j-1个tag的id（保存最大概率是从上一个时间点的哪个tag转移过来的。）
        # 等解码的时候，我们用backpointer进行回溯，以求出最优路径
        back_pointer = torch.zeros(self.tag_size, seq_len).long()

        # self.Pi[i] 表示第一个字的标记为i的概率
        # Bt[word_id]表示字为word_id的时候，对应各个标记的概率
        # self.A.t()[tag_id]表示各个状态转移到tag_id对应的概率

        # 所以第一步为
        start_wordid = word_to_index.get(word_list[0], None)
        Bt = B.t()
        if start_wordid is None:
            # 如果字不再字典里，则假设状态的概率分布是均匀的
            bt = torch.log(torch.ones(self.tag_size) / self.tag_size)
        else:
            bt = Bt[start_wordid]
        viterbi[:, 0] = Pi + bt
        back_pointer[:, 0] = -1
        # 递推公式：
        # viterbi[tag_id, step] = max(viterbi[:, step-1]* self.A.t()[tag_id] * Bt[word])
        # 其中word是step时刻对应的字
        # 由上述递推公式求后续各步
        for step in range(1, seq_len):
            wordid = word_to_index.get(word_list[step], None)
            # 处理字不在字典中的情况
            # bt是在t时刻字为wordid时，状态的概率分布
            if wordid is None:
                # 如果字不再字典里，则假设状态的概率分布是均匀的
                bt = torch.log(torch.ones(self.tag_size) / self.tag_size)
            else:
                bt = Bt[wordid]  # 否则从观测概率矩阵中取bt
            for tag_id in range(len(tag_to_index)):
                max_prob, max_id = torch.max(
                    viterbi[:, step-1] + A[:, tag_id],
                    dim=0
                )
                viterbi[tag_id, step] = max_prob + bt[tag_id]
                back_pointer[tag_id, step] = max_id
        # 终止， t=seq_len 即 viterbi[:, seq_len]中的最大概率，就是最优路径的概率
        best_path_prob, best_path_pointer = torch.max(
            viterbi[:, seq_len-1], dim=0
        )
        # 回溯，求最优路径
        best_path_pointer = best_path_pointer.item()
        best_path = [best_path_pointer]
        for back_step in range(seq_len-1, 0, -1):
            best_path_pointer = back_pointer[best_path_pointer, back_step]
            best_path_pointer = best_path_pointer.item()
            best_path.append(best_path_pointer)
        # 将tag_id组成的序列转化为tag
        assert len(best_path) == len(word_list)
        id2tag = dict((id_, tag) for tag, id_ in tag_to_index.items())
        tag_list = [id2tag[id_] for id_ in reversed(best_path)]
        return tag_list
