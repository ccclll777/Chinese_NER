import argparse
import torch
from data import DataSet
from models.crf import CRFModel
import os
from models.hmm import HMM
import datetime
from utils import  save_model,load_model
from evaluate import ConfusionMatrix
from models.bilstm_crf import BILSTM_Model
from models.transformer_crf import TransformerCRF_Model
from torch.utils.tensorboard import SummaryWriter
base_dir = os.getcwd()
def get_args():
    parser = argparse.ArgumentParser()
    #[hmm,crf,bilstm-crf,transformer-crf】
    parser.add_argument('--algorithm', type=str, default="bilstm-crf",choices=["hmm","crf","bilstm-crf","transformer-crf"])  # 用于选择哪种算法
    #['msra','clue'】
    parser.add_argument('--data-set', type=str, default="clue")  # 用于选择那个数据集
    parser.add_argument('--min-freq', type=int, default=10)  # 去掉频率小于min-freq的字
    """
    bilist crf 参数
    """
    #是否使用预训练的bert进行embedding
    parser.add_argument('--use-bert', action="store_true")
    parser.add_argument('--bert-model-dir', type=str, default="/bert/768")
    parser.add_argument('--embedding-size', type=int, default=768) #bert bilstm用768
    parser.add_argument('--hidden-size', type=int, default=384)  #bert bilstm用384
    parser.add_argument('--num-layers', type=int, default=2)  #lstm层数
    """
    transformer 参数
    """
    parser.add_argument('--d-model', type=int, default=128)#transformer model的维度 论文是512
    parser.add_argument('--num-blocks', type=int, default=2)#transformer block的数量
    parser.add_argument('--num-heads', type=int, default=4) #attention的head数
    parser.add_argument('--feedforward-dim', type=int, default=512)  #feedforward的隐藏层dim
    """
    训练参数
    """
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lr-min', type=float, default=0.00001)
    parser.add_argument('--use-dropout', action="store_true")
    parser.add_argument('--use-norm', action="store_true") #是否使用归一化层
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--grad-norm', default=5.0, type=float) #梯度裁剪
    parser.add_argument('--use-grad-norm', action="store_true")


    parser.add_argument('--log-step', type=int, default=5)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gpu-no', type=int, default=0)
    parser.add_argument('--logs', type=str, default='logs/')
    parser.add_argument('--save-model', action="store_false")
    parser.add_argument('--model-path', type=str, default="checkpoints/")
    """
    测试模型
    """
    parser.add_argument('--test-model-path', type=str, default="/checkpoints/clue/bilstm-crf/epoch_0.pth")
    parser.add_argument('--test', action="store_true")
    args = parser.parse_known_args()[0]
    return args

if __name__ == '__main__':
    args = get_args()
    print(args)
    data_set = DataSet(args.data_set)
    data_set.word_to_id(args.min_freq )
    args.bert_model_dir = base_dir+args.bert_model_dir
    args.model_path = args.model_path+args.data_set +"/"
    args.logs = args.logs+args.data_set +"/"
    if args.test == False:
        """
        训练模型
        """
        t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
        """
        模型保存相关
        """
        if args.use_bert == True:
            args.model_path = os.path.join(base_dir,args.model_path,args.algorithm+"-bert",t0)
        else:
            args.model_path = os.path.join(base_dir, args.model_path, args.algorithm, t0)
        folder = os.path.exists(args.model_path)
        if not folder and args.test == False:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(args.model_path)
        """
        log相关
        """
        if args.use_bert == True:
            args.logs = os.path.join(base_dir, args.logs, args.algorithm+"-bert", t0)
        else:
            args.logs = os.path.join(base_dir, args.logs, args.algorithm, t0)
        writer = SummaryWriter(args.logs)
        with open(args.logs+"/config.txt", 'w') as f:
            for key, value in vars(args).items():
                f.write('%s:%s\n' % (key, value))
        args.writer = writer
        if args.algorithm == "hmm":
            print("-------正在训练HMM模型-------")
            vocab_size = len(data_set.word_to_index)
            out_size = len(data_set.tag_to_index)
            x_train = data_set.x_train
            y_train = data_set.y_train
            hmm_model = HMM(out_size, vocab_size,data_set.word_to_index,data_set.tag_to_index)
            hmm_model.train(x_train,
                            y_train )
            save_model(hmm_model, args.model_path+"/hmm.pkl")
            print("-------正在评估HMM模型-------")
            # 评估hmm模型
            x_valid = data_set.x_valid
            pred_tag_lists = hmm_model.test(x_valid)
            y_valid = data_set.y_valid  # 真实tag
            confusion_matrix = ConfusionMatrix(y_valid, pred_tag_lists,data_set.tag_list)
            "打印评估结果"
            confusion_matrix.print_scores()
        elif args.algorithm == "crf":
            print("-------正在训练CRF模型-------")
            crf_model = CRFModel()
            x_train = data_set.x_train
            y_train = data_set.y_train
            crf_model.train(x_train, y_train)
            save_model(crf_model, args.model_path+"/crf.pkl")
            print("-------正在评估CRF模型-------")
            x_valid = data_set.x_valid
            predict__tag_lists = crf_model.test(x_valid) #预测tag
            y_valid = data_set.y_valid #真实tag
            #计算混淆矩阵
            confusion_matrix = ConfusionMatrix(y_valid, predict__tag_lists,data_set.tag_list)
            "打印评估结果"
            confusion_matrix.print_scores()
            # print("------混淆矩阵为---------")
            # confusion_matrix.report_confusion_matrix()
        elif args.algorithm == "bilstm-crf":
            #如果是bilstm-crf 还要加入<START>和<END> (解码的时候需要用到)
            data_set.extend_maps()
            vocab_size = len(data_set.word_to_index)
            out_size = len(data_set.tag_to_index)
            bilstm_model = BILSTM_Model(args,data_set)
            bilstm_model.train()
        elif args.algorithm == "transformer-crf":
            data_set.extend_maps()
            vocab_size = len(data_set.word_to_index)
            out_size = len(data_set.tag_to_index)
            transformer_model = TransformerCRF_Model(args,data_set)
            transformer_model.train()


    else:
        args.test_model_path = base_dir + args.test_model_path
        """
        测试模型
        """
        if args.algorithm == "hmm":
            hmm_model = load_model(args.test_model_path)
            x_valid = data_set.x_valid
            crf_pred = hmm_model.test(x_valid)
            y_valid = data_set.y_valid  # 真实tag
            confusion_matrix = ConfusionMatrix(y_valid, crf_pred,data_set.tag_list)
            "打印评估结果"
            confusion_matrix.print_scores()
        elif args.algorithm == "crf":
            print("-------正在评估CRF模型-------")
            crf_model = load_model(args.test_model_path)
            x_valid = data_set.x_valid
            crf_pred = crf_model.test(x_valid)
            y_valid = data_set.y_valid  # 真实tag
            confusion_matrix = ConfusionMatrix(y_valid, crf_pred,data_set.tag_list)
            "打印评估结果"
            confusion_matrix.print_scores()
            # print("------混淆矩阵为---------")
            # confusion_matrix.report_confusion_matrix()
        elif args.algorithm == "bilstm-crf":
            #如果是bilstm-crf 还要加入<START>和<END> (解码的时候需要用到)
            data_set.extend_maps()
            bilstm_model = BILSTM_Model(args,data_set)
            print("评估bilstm-crf模型中...")
            bilstm_model.model.load_state_dict(torch.load(args.test_model_path, map_location=args.device))
            pred_tag_lists, test_tag_lists = bilstm_model.test()
            confusion_matrix = ConfusionMatrix(test_tag_lists, pred_tag_lists,data_set.tag_list)
            "打印评估结果"
            confusion_matrix.print_scores()
            # print("------混淆矩阵为---------")
            # confusion_matrix.report_confusion_matrix()
        elif args.algorithm == "transformer-crf":
            #如果是bilstm-crf 还要加入<START>和<END> (解码的时候需要用到)
            data_set.extend_maps()
            transformer_model = TransformerCRF_Model(args,data_set)
            print("评估bilstm-crf模型中...")
            transformer_model.model.load_state_dict(torch.load(args.test_model_path, map_location=args.device))
            pred_tag_lists, test_tag_lists = transformer_model.test()
            confusion_matrix = ConfusionMatrix(test_tag_lists, pred_tag_lists,data_set.tag_list)
            "打印评估结果"
            confusion_matrix.print_scores()


