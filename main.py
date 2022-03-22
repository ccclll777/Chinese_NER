import argparse
import torch
from data import DataSet
from models.crf import CRFModel
import os
import datetime
from utils import  save_model
from evaluate import ConfusionMatrix
base_dir = os.getcwd()
def get_args():
    parser = argparse.ArgumentParser()
    #['crf'】
    parser.add_argument('--algorithm', type=str, default="crf")  # 用于选择哪种算法
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gpu-no', type=int, default=0)
    parser.add_argument('--logs', type=str, default='logs')
    parser.add_argument('--save-model', type=bool, default=True)
    parser.add_argument('--model-path', type=str, default="checkpoints")
    parser.add_argument('--test', type=bool, default=False)
    args = parser.parse_known_args()[0]
    return args

if __name__ == '__main__':
    args = get_args()
    data_set = DataSet()
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    args.model_path = os.path.join(base_dir,args.model_path,args.algorithm,t0)

    if args.algorithm == "crf":
        print("-------正在训练CRF模型-------")
        crf_model = CRFModel()
        x_train = data_set.decode_sentence_list(data_set.x_train.tolist())
        y_train = data_set.decode_tag_list(data_set.y_train.tolist())
        crf_model.train(x_train, y_train)
        save_model(crf_model, args.model_path+"_crf.pkl")
        print("-------正在评估CRF模型-------")
        x_valid = data_set.decode_sentence_list(data_set.x_valid.tolist())
        predict__tag_lists = crf_model.test(x_valid) #预测tag
        y_valid = data_set.decode_tag_list(data_set.y_valid.tolist()) #真实tag
        #计算混淆矩阵
        confusion_matrix = ConfusionMatrix(y_valid, predict__tag_lists)
        "打印评估结果"
        confusion_matrix.print_scores()
        print("------混淆矩阵为---------")
        confusion_matrix.report_confusion_matrix()



