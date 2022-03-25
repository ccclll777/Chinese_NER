import argparse
import torch
from data import DataSet
from models.crf import CRFModel
import os
import datetime
from utils import  save_model,load_model
from evaluate import ConfusionMatrix
from models.bilstm_crf import BILSTM_Model
from torch.utils.tensorboard import SummaryWriter
base_dir = os.getcwd()
def get_args():
    parser = argparse.ArgumentParser()
    #[crf,bilstm-crf】
    parser.add_argument('--algorithm', type=str, default="bilstm-crf")  # 用于选择哪种算法
    #['msra','clue'】
    parser.add_argument('--data-set', type=str, default="clue")  # 用于选择那个数据集
    """
    bilist crf 参数
    """
    #是否使用预训练的bert进行embedding
    parser.add_argument('--use-bert', type=bool, default=True)
    parser.add_argument('--bert-model-dir', type=str, default=base_dir+"/bert")
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=50)



    parser.add_argument('--log-step', type=int, default=5)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gpu-no', type=int, default=0)
    parser.add_argument('--logs', type=str, default='logs/')
    parser.add_argument('--save-model', type=bool, default=True)
    parser.add_argument('--model-path', type=str, default="checkpoints/")
    """
    测试模型
    """
    parser.add_argument('--test-model-path', type=str, default="/checkpoints/clue/bilstm-crf/0324_153734/epoch_8.pth")
    parser.add_argument('--test', type=bool, default=False)
    args = parser.parse_known_args()[0]
    return args

if __name__ == '__main__':
    args = get_args()
    data_set = DataSet(args.data_set)
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
        writer.add_text("args", str(args))
        args.writer = writer

        if args.algorithm == "crf":
            print("-------正在训练CRF模型-------")
            crf_model = CRFModel()
            x_train = data_set.x_train
            y_train = data_set.y_train
            crf_model.train(x_train, y_train)
            save_model(crf_model, args.model_path+"_crf.pkl")
            print("-------正在评估CRF模型-------")
            x_valid = data_set.x_valid
            predict__tag_lists = crf_model.test(x_valid) #预测tag
            y_valid = data_set.y_valid #真实tag
            #计算混淆矩阵
            confusion_matrix = ConfusionMatrix(y_valid, predict__tag_lists)
            "打印评估结果"
            confusion_matrix.print_scores()
            print("------混淆矩阵为---------")
            confusion_matrix.report_confusion_matrix()
        elif args.algorithm == "bilstm-crf":
            #如果是bilstm-crf 还要加入<START>和<END> (解码的时候需要用到)
            data_set.extend_maps()
            vocab_size = len(data_set.word_to_index)
            out_size = len(data_set.tag_to_index)
            if args.use_bert == True:
                args.embedding_size = 768
            bilstm_model = BILSTM_Model(args,data_set)
            bilstm_model.train()

    else:
        args.test_model_path = base_dir + args.test_model_path
        """
        测试模型
        """
        if args.algorithm == "crf":
            print("-------正在评估CRF模型-------")
            crf_model = load_model(args.test_model_path)
            x_valid = data_set.x_valid
            crf_pred = crf_model.test(x_valid)
            y_valid = data_set.y_valid  # 真实tag
            confusion_matrix = ConfusionMatrix(y_valid, crf_pred)
            "打印评估结果"
            confusion_matrix.print_scores()
            print("------混淆矩阵为---------")
            confusion_matrix.report_confusion_matrix()
        elif args.algorithm == "bilstm-crf":
            #如果是bilstm-crf 还要加入<START>和<END> (解码的时候需要用到)
            data_set.extend_maps()
            bilstm_model = BILSTM_Model(args,data_set)
            print("评估bilstm-crf模型中...")
            bilstm_model.model.load_state_dict(torch.load(args.test_model_path, map_location=args.device))
            pred_tag_lists, test_tag_lists = bilstm_model.test()
            confusion_matrix = ConfusionMatrix(test_tag_lists, pred_tag_lists)
            "打印评估结果"
            confusion_matrix.print_scores()
            print("------混淆矩阵为---------")
            confusion_matrix.report_confusion_matrix()


