import torch
from torch import nn
from itertools import zip_longest
class CRF(nn.Module):
    def __init__(self,out_size,device):
        """初始化参数：
        BiLSTM-CRF其实就是一个CRF模型，只不过用BiLSTM得到状态特征值sk，用反向传播算法更新转移特征值tk 。
            vocab_size:字典的大小
            embedding_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(CRF, self).__init__()
        self.device = device
        """
        CRF多学习一个转移矩阵 [out_size, out_size] 初始化为均匀分布 
        这个矩阵的ij表示从第i个tag转移到第j个tag的概率
        """
        self.transition = nn.Parameter(
            torch.ones(out_size, out_size) * 1/out_size)

    def forward(self, emission):
        """
        求出了每一帧对应到每种tag的发射矩阵 也就是每个字对应发射到每一种tag的概率 但是相加和不为1
        LSTM的输出——sentence的每个word经BiLSTM后，对应于每个tag的得分
        :param batch_sentences:
        :param sentence_lengths:
        :return:
        """
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

    def viterbi_decode(self, emission, lengths, tag_to_index):
        """
        使用维特比算法进行解码
        CRF层将BiLSTM的Emission_score作为输入，输出符合标注转移约束条件的、最大可能的预测标注序列。
        :param emission: BiLSTM的Emission_score作为输入
        :param lengths: 每句话的长度
        :param tag_to_id:
        :return:
        """
        """使用维特比算法进行解码"""
        start_id = tag_to_index['<CLS>']
        end_id = tag_to_index['<SEP>']
        pad = tag_to_index['<PAD>']
        tag_set_size = len(tag_to_index)
        crf_scores = self.forward(emission)#batch_size *length*output_size*output_size
        batch_size, max_len, target_set_size, _ = crf_scores.size()
        # viterbi[i, j, k]表示第i个句子，第j个字对应第k个tag的最大分数
        viterbi = torch.zeros(batch_size, max_len, target_set_size).to(self.device)#batch_size *length*output_size
        # back_point[i, j, k]表示第i个句子，第j个字对应第k个tag时前一个tag的id，用于回溯
        #找到得分最高的那个tag的id
        back_point = (torch.zeros(batch_size, max_len, target_set_size).long() * end_id).to(self.device)#batch_size *length*output_size
        lengths = torch.LongTensor(lengths).to(self.device)
        # 向前递推 遍历每一个字 进行推理
        for step in range(max_len):
            batch_size_t = (lengths > step).sum().item() #去除已经到达他们最大长度的那些句子
            if step == 0:
                # 第一个字它的前一个标记只能是start_id
                viterbi[:batch_size_t, step,:] = crf_scores[: batch_size_t, step, start_id, :]
                back_point[: batch_size_t, step, :] = start_id #刚开始初始化得分最高的tag的id为start
            else:
                pre_viterbi = viterbi[:batch_size_t, step-1, :].unsqueeze(2) #第step-1个字的对应每个tag的概率 word *tagsize*1 求最大概率的tag
                score = crf_scores[:batch_size_t, step, :, :]#第step个字从某个tag转移到另一个tag的得分word *tagsize*tagsize 最大概率的tag的得分
                max_scores, previous_tags = torch.max(
                    pre_viterbi + score,     # [batch_size, max_len, max_len]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores #word *tagsize
                back_point[:batch_size_t, step, :] = previous_tags  # word *tagsize

        # 在回溯的时候我们只需要用到back_point矩阵
        back_point = back_point.view(batch_size, -1)  # [batch_size,max_len* target_set_size]
        tag_ids = []  # 存放结果
        tags_t = None
        for step in range(max_len-1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == max_len-1:
                index = torch.ones(batch_size_t).long() * (step * tag_set_size)
                index = index.to(self.device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)

                new_in_batch = torch.LongTensor(
                    [end_id] * (batch_size_t - prev_batch_size_t)).to(self.device)
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )  # 这个offset实际上就是前一时刻的
                index = torch.ones(batch_size_t).long() * (step * tag_set_size)
                index = index.to(self.device)
                index += offset.long()

            try:
                tags_t = back_point[:batch_size_t].gather(
                    dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tag_ids.append(tags_t.tolist())

        # tag_ids:[max_len-1]（max_len-1是因为扣去了end_token),大小的liebiao
        # 其中列表内的元素是该batch在该时刻的标记
        # 下面修正其顺序，并将维度转换为 [batch_size, max_len]
        tag_ids = list(zip_longest(*reversed(tag_ids), fillvalue=pad))
        tag_ids = torch.Tensor(tag_ids).long()
        # 返回解码的结果
        return tag_ids
    def indexed(self,target_tags, tag_set_size, start_id):
        """
        将targets中的数转化为在[T*T]大小序列中的索引,T是标注的种类
        :param target_tags:
        :param tag_set_size:
        :param start_id:
        :return:
        """
        batch_size, max_len = target_tags.size()
        for col in range(max_len - 1, 0, -1):  # 从后向前遍历
            target_tags[:, col] += (target_tags[:, col - 1] * tag_set_size)
        target_tags[:, 0] += (start_id * tag_set_size)
        return target_tags
    def cal_lstm_crf_loss(self,crf_scores, target_tags,tag_to_index):
        """
                计算双向LSTM-CRF模型的损失
        该损失函数的计算可以参考:https://arxiv.org/pdf/1603.01360.pdf
        :param crf_scores:  序列中每个字符的Emission Score 和转移矩阵的拼接
                            [batch_size, max_len, out_size, out_size]
        :param target_tags:  tags
        :return:
        """
        pad_id = tag_to_index.get('<PAD>')
        start_id = tag_to_index.get('<CLS>')
        end_id = tag_to_index.get('<SEP>')

        # targets:[batch, max_len] crf_scores:[batch_size, max_len, out_size, out_size]
        batch_size, max_len = target_tags.size()
        target_size = len(tag_to_index) #tag的个数

        """
        使用crf_scores（发射矩阵+转移矩阵的拼接）；tags——真实序列标注，以此确定转移矩阵中选择哪条路径
        计算golden score  也就是真实的路径的score
        """
        # mask = 1 - ((targets == pad_id) + (targets == end_id))  # [B, L]
        mask = (target_tags != pad_id) #去掉所有的PAD PAD不参加计算
        lengths = mask.sum(dim=1) #每个句子的长度 有多少个单词
        target_tags = self.indexed(target_tags, target_size, start_id) #将tag转化成对应的index

        target_tags = target_tags.masked_select(mask)  # 将有PAD的位置去掉，只保留真实tag 维度为real_len

        flatten_scores = crf_scores.masked_select(
            mask.view(batch_size, max_len, 1, 1).expand_as(crf_scores)
        ).view(-1, target_size * target_size).contiguous()
        index = target_tags.unsqueeze(1) #找到真实路径的tag对应的index [real_len,1]

        # 筛选 我们需要的位置的值，然后求和，计算出真实路径得分
        # 对于每一个[output_size,output_siz]的矩阵， 选择其中的某一个位置，真实tag的位置
        golden_scores = flatten_scores.gather(
            dim=1, index=index) #[real_len,1]
        golden_scores = golden_scores.sum() #将所有real_len个值求和
        """
        计算所有路径的scores 
        输入 发射矩阵(emission score)， 输出：所有可能路径得分之和/归一化因子/配分函数/Z(x)
        https://www.yanxishe.com/columnDetail/21153
        """
        # 计算all path scores
        # scores_upto_t[i, j]表示第i个句子的第t个词被标注为j标记的所有t时刻事前的所有子路径的分数之和
        #每个时刻t的某个tag  会有tag_size个指向它的路径
        scores_upto_t = torch.zeros(batch_size, target_size).to(self.device)
        for t in range(max_len):
            # 当前时刻 有效的 batch_size（因为有些序列比较短) 可能在一些步骤之后 就不参与计算了
            batch_size_t = (lengths > t).sum().item()
            if t == 0: # start的得分
                scores_upto_t[:batch_size_t] = crf_scores[:batch_size_t,
                                               t, start_id, :]
            else:
                """
                 s crf_scores[:batch_size_t, t, :, :]：t时刻tag_i emission score（1个）的广播。
                                                        需要将其与t-1时刻的5个previous_tags转移到该tag_i
                                                        的transition scors相加
                 cores_upto_t[:batch_size_t].unsqueeze(2)：t-1时刻的5个previous_tags到该tag_i的
                                                                transition scors
                https://www.yanxishe.com/columnDetail/21153
                """
                scores_upto_t[:batch_size_t] = torch.logsumexp(
                    crf_scores[:batch_size_t, t, :, :] +
                    scores_upto_t[:batch_size_t].unsqueeze(2),
                    dim=1 )
        all_path_scores = scores_upto_t[:, end_id].sum()
        # 训练大约两个epoch loss变成负数，从数学的角度上来说，loss = -logP
        loss = (all_path_scores - golden_scores) / batch_size
        return loss
