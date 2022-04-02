from collections import Counter

import itertools
from prettytable import PrettyTable

class ConfusionMatrix(object):
    """计算混淆矩阵"""
    def __init__(self, tags, predict_tags,tags_list):
        """

        :param tags:  真实标签
        :param predict_tags:  预测标签
        :param tags_list: 所有的标签列表
        """

        # 将tag的list 展平  [[xxx],[],[],[],[xxx]]->[xxxxxxxxxxxx]
        #展平操作 可以用来对比每一个tag是否能对应
        self.tags = list(itertools.chain.from_iterable(tags))
        self.predict_tags = list(itertools.chain.from_iterable(predict_tags))
        # 所有tag的set
        self.tags_list = tags_list
        self.tag_set = set(self.tags_list)
        # 计算每一种标记正确的个数 对应TP  返回dict
        self.correct_tags_count = self.get_correct_tag()
        # self.correct_dict = {"address":0,"book":0,"company":0,"address":0,"address":0,"address":0,"address":0,"address":0}
        # 计算predict_tags中每一个tag的出现次数
        # self.predict_tags_count = Counter(self.predict_tags)
        # 计算label tag中每一个tag的出现次数
        self.gold_tags_count,self.predict_tags_count = self.get_tag()
        # self.tags_count = Counter(self.tags)
        # 计算精确率 TP/TP+FP 为正例的示例中实际为正例的比例
        self.precision_scores = self.get_precision()
        # 计算召回率 TP/(TP+FN) 多个正例被分为正例
        self.recall_scores = self.get_recall()
        # 计算F1分数
        self.f1_scores = self.get_f1()

    def get_precision(self):
        """
        计算精确率 TP/TP+FP 为正例的示例中实际为正例的比例
        将每个位置 每个tag分别计算精确率
        分母为应该为某个tag的数量
        :return:
        """
        precision_scores = {}
        for tag in self.tag_set:
            precision_scores[tag] = self.correct_tags_count.get(tag, 0) /  (self.predict_tags_count[tag]+1e-10)
        return precision_scores
    def get_recall(self):
        """
        计算召回率 TP/(TP+FN) 多个正例被分为正例
        分母为预测为某个tag的数量
        :return:
        """
        recall_scores = {}
        for tag in self.tag_set:
            recall_scores[tag] = self.correct_tags_count.get(tag, 0) / (self.gold_tags_count[tag] +1e-10)
        return recall_scores

    def get_f1(self):
        f1_scores = {}
        for tag in self.tag_set:
            precision, recall = self.precision_scores[tag], self.recall_scores[tag]
            f1_scores[tag] = 2*precision*recall / (recall+precision+1e-10)
        return f1_scores
    def get_correct_tag(self):
        """
        计算每种标签预测正确的个数(对应精确率、召回率计算公式上的tp)，用于后面精确率以及召回率的计算
        #计算每一种标记正确的个数 对应tp 返回dict
        :return:
        """
        correct_dict = {}
        for i in self.tags_list:
            correct_dict[i] = 0
        for gold_tag, predict_tag in zip(self.tags, self.predict_tags):
            if gold_tag == predict_tag:
                if gold_tag == "o":
                    correct_dict[gold_tag] +=1
                else:
                    correct_dict[gold_tag[2:]] += 1
        return correct_dict
    def get_tag(self):
        """
        计算真实tag中每一类的个数  计算预测tag中每一类的个数
        :return:
        """
        gold_tag_dict = {}
        predict_tag_dict = {}
        for i in self.tags_list:
            gold_tag_dict[i] = 0
            predict_tag_dict[i] = 0
        for gold_tag in self.tags:
                if gold_tag == "o":
                    gold_tag_dict[gold_tag] +=1
                elif gold_tag != "<PAD>" and gold_tag != "<SEP>" and gold_tag != "<CLS>" and gold_tag != "<UNK>":
                    gold_tag_dict[gold_tag[2:]] += 1
        for predict_tag in self.predict_tags:
                if predict_tag == "o":
                    predict_tag_dict[predict_tag] +=1
                elif predict_tag != "<PAD>" and predict_tag != "<SEP>" and predict_tag != "<CLS>" and predict_tag != "<UNK>":
                    predict_tag_dict[predict_tag[2:]] += 1
        return gold_tag_dict,predict_tag_dict
    def get_average(self):

        average = {}
        total = len(self.tags)
        # 计算weighted precisions:
        average['precision'] = 0.
        average['recall'] = 0.
        average['f1_score'] = 0.
        for tag in self.tag_set:
            size = self.gold_tags_count[tag]
            average['precision'] += self.precision_scores[tag] * size
            average['recall'] += self.recall_scores[tag] * size
            average['f1_score'] += self.f1_scores[tag] * size

        for metric in average.keys():
            average[metric] /= total
        return average
    def print_scores(self):
        """
        将结果用表格的形式打印
        :return:
        """
        # 表头
        table = PrettyTable(['    ', 'precision', 'recall', 'f1-score',"tag_count"])
        # 打印每个标签的 精确率、召回率、f1分数
        for tag in self.tag_set:
            table.add_row([tag, str(self.precision_scores[tag]),
                           str(self.recall_scores[tag]), str(self.f1_scores[tag]),
                           str(self.gold_tags_count[tag])])


        # 计算并打印平均值
        # average = self.get_average()
        # table.add_row(['avg/total', str( average['precision']),
        #                    str(average['recall']), str(average['f1_score']),
        #                    str(len(self.tags))])
        print(table)
        # return self.precision_scores,self.recall_scores,self.f1_scores

    def report_confusion_matrix(self):
        """
        计算混淆矩阵
        :return:
        """
        tag_list = list(self.tag_set)
        # 初始化混淆矩阵 matrix[i][j]表示第i个tag被模型预测成第j个tag的次数
        tags_size = len(tag_list)
        matrix = []
        for i in range(tags_size):
            matrix.append([0] * tags_size)

        # 遍历tags列表
        for tag, predict_tag in zip(self.tags, self.predict_tags):
            try:
                row = tag_list.index(tag)
                col = tag_list.index(predict_tag)
                matrix[row][col] += 1
            except ValueError:  # 有极少数标记没有出现在golden_tags，但出现在predict_tags，跳过这些标记
                continue
        # 输出矩阵
        table = PrettyTable(['    ', tag_list[0],tag_list[1], tag_list[2], tag_list[3],
                             tag_list[4], tag_list[5], tag_list[6], tag_list[7], tag_list[8], tag_list[9]])
        for i, row in enumerate(matrix):
            table.add_row(
                [tag_list[i], str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]),
                 str(row[5]),str(row[6]), str(row[7]), str(row[8]), str(row[9])])
        print(table)