from itertools import zip_longest
from copy import deepcopy
import time
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils import sort_by_lengths
from networks.bilstm_crf_network import BiLSTM_CRF
class BILSTM_Model(object):
    def __init__(self, args,data_set):
        """功能：对LSTM的模型进行训练与测试
           参数:
            vocab_size:词典大小
            out_size:标注种类
        """
        if args.test == False:
            self.writer = args.writer
        else:
            self.writer = None
        self.args = args
        self.data_set = data_set
        self.vocab_size = len(data_set.word_to_index)
        self.out_size = len(data_set.tag_to_index)
        self.device = args.device
        # 加载模型参数
        self.embedding_size = args.embedding_size
        self.hidden_size =  args.hidden_size
        # 根据是否添加crf初始化不同的模型 选择不一样的损失计算函数
        self.model = BiLSTM_CRF(self.vocab_size, self.embedding_size,
                                    self.hidden_size, self.out_size,self.device,args.use_bert ,args.bert_model_dir ).to(self.device)
        # self.cal_loss_func = cal_lstm_crf_loss
        # 训练参数：
        self.epoch = args.epoch
        self.log_step = args.log_step
        self.lr = args.lr
        self.batch_size = args.batch_size
        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 初始化其他指标
        self.step = 0
        self.train_log_step =0
        self.best_val_loss = 1e18

    def cal_lstm_crf_loss(self,crf_scores, target_tags,):
        """
                计算双向LSTM-CRF模型的损失
        该损失函数的计算可以参考:https://arxiv.org/pdf/1603.01360.pdf
        :param crf_scores:  序列中每个字符的Emission Score 和转移矩阵的拼接
                            [batch_size, max_len, out_size, out_size]
        :param target_tags:  tags
        :return:
        """
        pad_id = self.data_set.tag_to_index.get('<PAD>')
        start_id = self.data_set.tag_to_index.get('<START>')
        end_id = self.data_set.tag_to_index.get('<END>')

        # targets:[batch, max_len] crf_scores:[batch_size, max_len, out_size, out_size]
        batch_size, max_len = target_tags.size()
        target_size = len(self.data_set.tag_to_index) #tag的个数

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

    def train(self):
        """
        训练 bilstm crf
        :return:
        """
        temp_train_word_lists, temp_train_tag_lists = self.data_set.prepocess_data_for_lstm_crf(
            self.data_set.x_train, self.data_set.y_train
        )
        temp_valid_word_lists, temp_valid_tag_lists = self.data_set.prepocess_data_for_lstm_crf(
            self.data_set.x_valid, self.data_set.y_valid
        )

        #数据集按长度排序
        train_word_lists, train_tag_lists, _ = sort_by_lengths(temp_train_word_lists, temp_train_tag_lists)
        valid_word_lists, valid_tag_lists, _ = sort_by_lengths(temp_valid_word_lists, temp_valid_tag_lists)
        for i in range(self.epoch):
            self.step = 0
            losses = 0
            #按照batch进行训练
            for index in range(0, len(train_word_lists), self.batch_size):
                #切分出一个batch的数据
                batch_sents = train_word_lists[index:index+self.batch_size]
                batch_tags = train_tag_lists[index:index+self.batch_size]
                losses += self.train_step(batch_sents,
                                          batch_tags)

                if self.step % self.log_step == 0:
                    total_step = (len(train_word_lists) // self.batch_size + 1)
                    print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                        i, self.step, total_step,
                        100. * self.step / total_step,
                        losses / self.log_step
                    ))
                    self.train_log_step+=1
                    self.writer.add_scalar("train/loss", losses / self.log_step, self.train_log_step)
                    losses = 0.

            # 每轮结束测试在验证集上的性能，保存最好的一个
            val_loss = self.validate(i,valid_word_lists, valid_tag_lists)
            print("Epoch {}, Val Loss:{:.4f}".format(i, val_loss))
            self.writer.add_scalar("val/loss", val_loss, i)
    def train_step(self, batch_sentences, batch_tags):
        """
        训练一个batch的数据
        :param batch_sentences:
        :param batch_tags:
        :return:
        """
        self.model.train()
        self.step += 1
        # 准备数据 将数据集转化为index
        token_sentences, lengths = self.data_set.bilstm_crf_word_to_index(batch_sentences,
                                            self.data_set.word_to_index)
        token_sentences = token_sentences.to(self.device)
        target_tags, lengths = self.data_set.bilstm_crf_word_to_index(batch_tags,
                                            self.data_set.tag_to_index)
        target_tags = target_tags.to(self.device)
        # forward
        scores = self.model(token_sentences, lengths)
        # 计算损失 更新参数
        self.optimizer.zero_grad()
        loss = self.cal_lstm_crf_loss(scores, target_tags).to(self.device)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def validate(self,epoch,valid_word_lists,valid_tag_lists):
        """
        评估模型
        :param epoch 当前的epoch
        :param valid_word_lists:
        :param valid_tag_lists:
        :return:
        """
        print("评估模型中～～～～～～")
        self.model.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for index in tqdm(range(0, len(valid_word_lists), self.batch_size)):
                val_step += 1
                # 准备batch数据
                batch_sentences = valid_word_lists[index:index+self.batch_size]
                batch_tags = valid_tag_lists[index:index+self.batch_size]
                #将数据转化为index
                valid_token_sentences, lengths = self.data_set.bilstm_crf_word_to_index(
                    batch_sentences,self.data_set.word_to_index)
                valid_token_sentences = valid_token_sentences.to(self.device)
                target_tags, lengths = self.data_set.bilstm_crf_word_to_index(batch_tags,
                                                                          self.data_set.tag_to_index)
                target_tags = target_tags.to(self.device)

                # forward
                scores = self.model(valid_token_sentences, lengths)

                # 计算损失
                loss = self.cal_lstm_crf_loss(
                    scores, target_tags).to(self.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step

            if val_loss < self.best_val_loss:
                print("保存模型...")
                torch.save(self.model.state_dict(),
                           self.args.model_path + "/epoch_"+str(epoch)+".pth")
                self.best_val_loss = val_loss

            return val_loss
    def test(self,):
        """
        测试模型
        :return:
        """
        # 准备数据
        temp_test_word_lists, temp_test_tag_lists = self.data_set.prepocess_data_for_lstm_crf(
            self.data_set.x_test, self.data_set.y_test, test=True
        )
        test_word_lists, test_tag_lists, indices = sort_by_lengths(temp_test_word_lists, temp_test_tag_lists)
        test_token_sentences, lengths = self.data_set.bilstm_crf_word_to_index(test_word_lists, self.data_set.word_to_index)
        test_token_sentences = test_token_sentences.to(self.device)

        self.model.eval()
        with torch.no_grad():
            batch_tagids = self.model.viterbi_decode(
                test_token_sentences, lengths,self.data_set.tag_to_index)

        # 将id转化为标注
        pred_tag_lists = []
        # id2tag = dict((id_, tag) for tag, id_ in self.data_set.tag_to_index.items())
        for i, ids in enumerate(batch_tagids):
            tag_list = []
            for j in range(lengths[i] - 1):  # crf解码过程中，end被舍弃
                tag_list.append(self.data_set.index_to_tag[ids[j].item()])
            pred_tag_lists.append(tag_list)

        # indices存有根据长度排序后的索引映射的信息
        # 比如若indices = [1, 2, 0] 则说明原先索引为1的元素映射到的新的索引是0，
        # 索引为2的元素映射到新的索引是1...
        # 下面根据indices将pred_tag_lists和tag_lists转化为原来的顺序
        ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
        indices, _ = list(zip(*ind_maps))
        pred_tag_lists = [pred_tag_lists[i] for i in indices]
        tag_lists = [test_tag_lists[i] for i in indices]
        return pred_tag_lists, tag_lists