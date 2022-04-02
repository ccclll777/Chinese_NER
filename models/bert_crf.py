import torch
from networks.bert_crf_network import BertCRF
from tqdm import tqdm
from utils import sort_by_lengths,build_optimizer_and_scheduler
class BertCRFModel(object):
    def __init__(self, args,data_set):
        """功能：对BertCRF的模型进行训练与测试
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
        self.num_tags = len(data_set.tag_to_index)
        self.device = args.device

        self.hidden_size =  args.hidden_size
        # 根据是否添加crf初始化不同的模型 选择不一样的损失计算函数
        self.model = BertCRF(hidden_size = self.hidden_size,
                 num_tags = self.num_tags,
                 # num_layers = args.num_layers,
                 device = self.device,
                 dropout = args.dropout,
                 bert_model_dir = args.bert_model_dir).to(args.device)
        # 训练参数：
        self.epoch = args.epoch
        self.log_step = args.log_step
        self.batch_size = args.batch_size
        self.grad_norm = args.grad_norm
        self.use_grad_norm = args.use_grad_norm
        # 初始化优化器 使用差分学习率
        self.t_total = len(self.data_set.x_train) * self.epoch
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(args, self.model, self.t_total)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, float(self.epoch), eta_min=args.lr_min)
        # 初始化其他指标
        self.step = 0
        self.train_log_step =0
        self.best_val_loss = 1e18

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
        train_word_lists, train_tag_lists, train_word_lengths = sort_by_lengths(temp_train_word_lists, temp_train_tag_lists)
        valid_word_lists, valid_tag_lists, valid_word_lengths = sort_by_lengths(temp_valid_word_lists, temp_valid_tag_lists)
        self.model.zero_grad()
        for i in range(self.epoch):
            self.model.train()
            self.step = 0
            losses = 0
            #按照batch进行训练
            for index in tqdm(range(0, len(train_word_lists), self.batch_size)):
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
        token_sentences, lengths,sentences_mask,token_type_ids = self.data_set.bert_crf_word_to_index(batch_sentences,
                                            self.data_set.word_to_index)
        token_sentences = token_sentences.to(self.device)
        sentences_mask = sentences_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device)

        target_tags, lengths,_,_ = self.data_set.bert_crf_word_to_index(batch_tags,
                                            self.data_set.tag_to_index)
        target_tags = target_tags.to(self.device)

        # forward
        scores = self.model(token_sentences, sentences_mask,token_type_ids)

        # 计算损失 更新参数
        # self.optimizer.zero_grad()
        loss = self.model.crf.cal_lstm_crf_loss(scores, target_tags,self.data_set.tag_to_index).to(self.device)

        if self.use_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.model.zero_grad()
        # self.optimizer.step()
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
                valid_token_sentences, lengths,sentences_mask,valid_token_type_ids= self.data_set.bert_crf_word_to_index(
                    batch_sentences,self.data_set.word_to_index)
                valid_token_sentences = valid_token_sentences.to(self.device)
                valid_token_type_ids = valid_token_type_ids.to(self.device)
                sentences_mask = sentences_mask.to(self.device)
                target_tags, lengths,_,_ = self.data_set.bert_crf_word_to_index(batch_tags,
                                                                          self.data_set.tag_to_index)
                target_tags = target_tags.to(self.device)

                # forward
                scores = self.model(valid_token_sentences, sentences_mask,valid_token_type_ids)
                # 计算损失
                loss = self.model.crf.cal_lstm_crf_loss(scores, target_tags, self.data_set.tag_to_index).to(self.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step

            if val_loss < self.best_val_loss:
                print("保存模型...")
                torch.save(self.model.state_dict(),
                           self.args.model_path + "/epoch_"+str(epoch)+".pth")
                self.best_val_loss = val_loss

            return val_loss
    def test(self,word_list,tag_list):
        """
        测试模型
        :return:
        """
        # 准备数据
        temp_test_word_lists, temp_test_tag_lists = self.data_set.prepocess_data_for_lstm_crf(word_list, tag_list, test=True)
        test_word_lists, test_tag_lists, indices = sort_by_lengths(temp_test_word_lists, temp_test_tag_lists)
        test_token_sentences, lengths,sentences_mask,test_token_type_ids = self.data_set.bert_crf_word_to_index(test_word_lists, self.data_set.word_to_index)
        test_token_sentences = test_token_sentences.to(self.device)
        sentences_mask = sentences_mask.to(self.device)
        test_token_type_ids = test_token_type_ids.to(self.device)

        self.model.eval()
        with torch.no_grad():
            crf_scores, batch_tagids = self.model.test(test_token_sentences,sentences_mask,test_token_type_ids, lengths,self.data_set.tag_to_index)
        # 将id转化为标注
        pred_tag_lists = []
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