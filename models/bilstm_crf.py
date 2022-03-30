from itertools import zip_longest
from copy import deepcopy
import time
import torch
from tqdm import tqdm
from evaluate import ConfusionMatrix
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
        self.model = BiLSTM_CRF(vocab_size=self.vocab_size,
                                embedding_size=self.embedding_size,
                                hidden_size=self.hidden_size,
                                out_size=self.out_size,
                                num_layers=args.num_layers,
                                device=self.device,
                                dropout=args.dropout,
                                use_dropout=args.use_dropout,
                                use_norm = args.use_norm,
                                use_bert=args.use_bert ,
                                fine_tuning = args.fine_tuning,
                                bert_model_dir =args.bert_model_dir).to(self.device)
        # 训练参数：
        self.epoch = args.epoch
        self.log_step = args.log_step
        self.lr = args.lr
        self.batch_size = args.batch_size
        #对梯度进行裁剪
        self.grad_norm = args.grad_norm
        self.use_grad_norm = args.use_grad_norm
        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, float(self.epoch), eta_min=args.lr_min)
        # 初始化其他指标
        self.step = 0
        self.train_log_step =0
        self.best_val_loss = 1e18
        self.model_path = None



    def train(self):
        """
        训练 bilstm crf
        :return:
        """
        temp_train_word_lists, temp_train_tag_lists = self.data_set.prepocess_data_for_lstm_crf(
            self.data_set.x_train, self.data_set.y_train
        )
        temp_valid_word_lists, temp_valid_tag_lists = self.data_set.prepocess_data_for_lstm_crf(
            self.data_set.x_valid, self.data_set.y_valid)

        #数据集按长度排序
        train_word_lists, train_tag_lists, _ = sort_by_lengths(temp_train_word_lists, temp_train_tag_lists)
        valid_word_lists, valid_tag_lists, _ = sort_by_lengths(temp_valid_word_lists, temp_valid_tag_lists)
        for i in range(self.epoch):
            self.model.train()
            self.step = 0
            losses = 0
            #按照batch进行训练
            for index in range(0, len(train_word_lists), self.batch_size):
                #切分出一个batch的数据
                batch_sents = train_word_lists[index:index+self.batch_size]
                batch_tags = train_tag_lists[index:index+self.batch_size]
                losses += self.train_step(batch_sents,
                                          batch_tags)
                self.scheduler.step()
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
            # print("self.data_set.x_test")
            # pred_tag_lists, test_tag_lists = self.test(self.data_set.x_test, self.data_set.y_test)
            # confusion_matrix = ConfusionMatrix(test_tag_lists, pred_tag_lists, self.data_set.tag_list)
            # "打印评估结果"
            # confusion_matrix.print_scores()
            # for tag in confusion_matrix.tag_set:
            #     self.writer.add_scalar("val/"+tag+"_precision", confusion_matrix.precision_scores[tag], i)
            #     self.writer.add_scalar("val/"+tag+"_recall", confusion_matrix.recall_scores[tag], i)
            #     self.writer.add_scalar("val/"+tag+"_f1", confusion_matrix.f1_scores[tag], i)
            # bilstm_model = BILSTM_Model(self.args,self.data_set)
            # print("评估bilstm-crf模型中...")
            # bilstm_model.model.load_state_dict(torch.load(self.model_path , map_location=self.args.device))
            # pred_tag_lists, test_tag_lists = bilstm_model.test(bilstm_model.data_set.x_valid,bilstm_model.data_set.y_valid)
            # confusion_matrix = ConfusionMatrix(test_tag_lists, pred_tag_lists,self.data_set.tag_list)
            # "打印评估结果"
            # confusion_matrix.print_scores()

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
        loss = self.model.crf.cal_lstm_crf_loss(scores, target_tags,self.data_set.tag_to_index).to(self.device)
        loss.backward()
        if self.use_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
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
            for index in range(0, len(valid_word_lists), self.batch_size):
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
                scores,pred_tag_ids = self.model.test(valid_token_sentences, lengths,self.data_set.tag_to_index)
                # 计算损失
                loss = self.model.crf.cal_lstm_crf_loss(
                    scores, target_tags,self.data_set.tag_to_index).to(self.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step

            if val_loss < self.best_val_loss:
                print("保存模型...")
                torch.save(self.model.state_dict(),
                           self.args.model_path + "/epoch_"+str(epoch)+".pth")
                self.model_path = self.args.model_path + "/epoch_"+str(epoch)+".pth"
                self.best_val_loss = val_loss

            return val_loss
    def test(self,word_list,label_list):
        """
        测试模型
        :return:
        """
        # 准备数据
        temp_test_word_lists, temp_test_tag_lists = self.data_set.prepocess_data_for_lstm_crf(
            word_list, label_list, test=True
        )
        test_word_lists, test_tag_lists, indices = sort_by_lengths(temp_test_word_lists, temp_test_tag_lists)
        test_token_sentences, lengths = self.data_set.bilstm_crf_word_to_index(test_word_lists, self.data_set.word_to_index)
        test_token_sentences = test_token_sentences.to(self.device)

        self.model.eval()
        with torch.no_grad():
            crf_scores, batch_tagids = self.model.test(
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