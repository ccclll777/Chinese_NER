import torch
from torch import nn
from networks.bilstm_network import BiLSTM
from itertools import zip_longest
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, out_size,device):
        """初始化参数：
        BiLSTM-CRF其实就是一个CRF模型，只不过用BiLSTM得到状态特征值sk，用反向传播算法更新转移特征值tk 。
            vocab_size:字典的大小
            embedding_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM_CRF, self).__init__()
        self.bilstm = BiLSTM(vocab_size, embedding_size, hidden_size, out_size)
        self.device = device
        """
        CRF多学习一个转移矩阵 [out_size, out_size] 初始化为均匀分布 
        这个矩阵的ij表示从第i个tag转移到第j个tag的概率
        """
        self.transition = nn.Parameter(
            torch.ones(out_size, out_size) * 1/out_size)

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
        batch_size, max_len, out_size = emission.size()
        #增加一个维度
        temp_transition = self.transition.unsqueeze(0) #[1,out_size,out_size]
        #emission.unsqueeze(2) [batch_size,max_len,1,out_size]
        #expand(-1, -1, out_size, -1) [batch_size,max_len,out_size,out_size]
        crf_scores = emission.unsqueeze(2).expand(-1, -1, out_size, -1) + temp_transition

        return crf_scores

    def viterbi_decode(self, test_batch_sentences, lengths, tag_to_index):
        """
        使用维特比算法进行解码
        CRF层将BiLSTM的Emission_score作为输入，输出符合标注转移约束条件的、最大可能的预测标注序列。
        :param test_batch_sentences:
        :param lengths:
        :param tag_to_id:
        :return:
        """
        """使用维特比算法进行解码"""
        start_id = tag_to_index['<START>']
        end_id = tag_to_index['<END>']
        pad = tag_to_index['<PAD>']
        tagset_size = len(tag_to_index)

        crf_scores = self.forward(test_batch_sentences, lengths)
        device = crf_scores.device
        # B:batch_size, L:max_len, T:target set size
        B, L, T, _ = crf_scores.size()
        # viterbi[i, j, k]表示第i个句子，第j个字对应第k个标记的最大分数
        viterbi = torch.zeros(B, L, T).to(device)
        # backpointer[i, j, k]表示第i个句子，第j个字对应第k个标记时前一个标记的id，用于回溯
        backpointer = (torch.zeros(B, L, T).long() * end_id).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        # 向前递推
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                # 第一个字它的前一个标记只能是start_id
                viterbi[:batch_size_t, step,
                        :] = crf_scores[: batch_size_t, step, start_id, :]
                backpointer[: batch_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step-1, :].unsqueeze(2) +
                    crf_scores[:batch_size_t, step, :, :],     # [B, T, T]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        # 在回溯的时候我们只需要用到backpointer矩阵
        backpointer = backpointer.view(B, -1)  # [B, L * T]
        tagids = []  # 存放结果
        tags_t = None
        for step in range(L-1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L-1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)

                new_in_batch = torch.LongTensor(
                    [end_id] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )  # 这个offset实际上就是前一时刻的
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += offset.long()

            try:
                tags_t = backpointer[:batch_size_t].gather(
                    dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())

        # tagids:[L-1]（L-1是因为扣去了end_token),大小的liebiao
        # 其中列表内的元素是该batch在该时刻的标记
        # 下面修正其顺序，并将维度转换为 [B, L]
        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
        tagids = torch.Tensor(tagids).long()

        # 返回解码的结果
        return tagids
        # start_id = tag_to_index['<START>']
        # end_id = tag_to_index['<END>']
        # pad = tag_to_index['<PAD>']
        # tag_set_size = len(tag_to_index)
        #
        # crf_scores = self.forward(test_batch_sentences, lengths)#batch_size *length*output_size*output_size
        #
        # batch_size, max_len, target_set_size, _ = crf_scores.size()
        # # viterbi[i, j, k]表示第i个句子，第j个字对应第k个tag的最大分数
        # viterbi = torch.zeros(batch_size, max_len, target_set_size).to(self.device)#batch_size *length*output_size
        # # back_point[i, j, k]表示第i个句子，第j个字对应第k个tag时前一个tag的id，用于回溯
        # #找到得分最高的那个tag的id
        # back_point = (torch.zeros(batch_size, max_len, target_set_size).long() * end_id).to(self.device)#batch_size *length*output_size
        # lengths = torch.LongTensor(lengths).to(self.device)
        # # 向前递推 遍历每一个字 进行推理
        # for step in range(max_len):
        #     batch_size_t = (lengths > step).sum().item() #去除已经到达他们最大长度的那些句子
        #     if step == 0:
        #         # 第一个字它的前一个标记只能是start_id
        #         viterbi[:batch_size_t, step,:] = crf_scores[: batch_size_t, step, start_id, :]
        #         back_point[: batch_size_t, step, :] = start_id #刚开始初始化得分最高的tag的id为start
        #     else:
        #         pre_viterbi = viterbi[:batch_size_t, step-1, :].unsqueeze(2) #第step-1个字的对应每个tag的概率 word *tagsize*1 求最大概率的tag
        #         score = crf_scores[:batch_size_t, step, :, :]#第step个字从某个tag转移到另一个tag的得分word *tagsize*tagsize 最大概率的tag的得分
        #         max_scores, previous_tags = torch.max(
        #             pre_viterbi + score,     # [batch_size, max_len, max_len]
        #             dim=1
        #         )
        #         viterbi[:batch_size_t, step, :] = max_scores #word *tagsize
        #         back_point[:batch_size_t, step, :] = previous_tags  # word *tagsize
        #
        # # 在回溯的时候我们只需要用到back_point矩阵
        # back_point = back_point.view(batch_size, -1)  # [batch_size,max_len* target_set_size]
        # tag_ids = []  # 存放结果
        # tags_t = None
        # for step in range(max_len-1, 0, -1):
        #     batch_size_t = (lengths > step).sum().item()
        #     if step == max_len-1:
        #         index = torch.ones(batch_size_t).long() * (step * tag_set_size)
        #         index = index.to(self.device)
        #         index += end_id
        #     else:
        #         prev_batch_size_t = len(tags_t)
        #
        #         new_in_batch = torch.LongTensor(
        #             [end_id] * (batch_size_t - prev_batch_size_t)).to(self.device)
        #         offset = torch.cat(
        #             [tags_t, new_in_batch],
        #             dim=0
        #         )  # 这个offset实际上就是前一时刻的
        #         index = torch.ones(batch_size_t).long() * (step * tag_set_size)
        #         index = index.to(self.device)
        #         index += offset.long()
        #
        #     try:
        #         tags_t = back_point[:batch_size_t].gather(
        #             dim=1, index=index.unsqueeze(1).long())
        #     except RuntimeError:
        #         import pdb
        #         pdb.set_trace()
        #     tags_t = tags_t.squeeze(1)
        #     tag_ids.append(tags_t.tolist())
        #
        # # tag_ids:[max_len-1]（max_len-1是因为扣去了end_token),大小的liebiao
        # # 其中列表内的元素是该batch在该时刻的标记
        # # 下面修正其顺序，并将维度转换为 [batch_size, max_len]
        # tag_ids = list(zip_longest(*reversed(tag_ids), fillvalue=pad))
        # tag_ids = torch.Tensor(tag_ids).long()
        #
        # # 返回解码的结果
        # return tag_ids
