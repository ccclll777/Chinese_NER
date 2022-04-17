from flask import Flask
from flask_cors import CORS
from flask_restful import Api
from flask_restful import Resource,reqparse,request
import sys
import os
base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
print(base_dir)
sys.path.append(base_dir)
from data import DataSet
import torch
import argparse
from models.bilstm_crf import BILSTM_Model
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-freq', type=int, default=10)  # 去掉频率小于min-freq的字
    """
    bilist crf 参数
    """
    #是否使用预训练的bert进行embedding
    parser.add_argument('--use-bert', action="store_true")
    parser.add_argument('--fine-tuning', action="store_true") #是否微调bert

    parser.add_argument('--bert-model-dir', type=str, default="/bert/bert-base-chinese")
    parser.add_argument('--embedding-size', type=int, default=128) #bert bilstm用768
    parser.add_argument('--hidden-size', type=int, default=256)  #bert bilstm用384
    parser.add_argument('--num-layers', type=int, default=2)  #lstm层数
    """
    差分学习率和warmup配置
    """
    # 2e-5
    parser.add_argument('--bert-lr', default=3e-5, type=float,
                        help='bert学习率')
    # 2e-3
    parser.add_argument('--lr', default=3e-4, type=float,help='bilstm学习率')
    parser.add_argument('--crf-lr', default=3e-2, type=float, help='crf学习率')
    parser.add_argument('--warmup-proportion', default=0.1, type=float,help= " warm up的步数比例x（全局总训练次数t中前x*t步）")

    parser.add_argument('--weight-decay', default=0.01, type=float,help="权重衰减的比例")

    parser.add_argument('--adam-epsilon', default=1e-8, type=float)
    """
    训练参数
    """
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--use-dropout', default="True")
    parser.add_argument('--use-norm', default="True") #是否使用归一化层
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--grad-norm', default=1.0, type=float) #梯度裁剪
    parser.add_argument('--use-grad-norm', action="store_true")
    parser.add_argument('--log-step', default=10)


    parser.add_argument('--sentence-ner',default="True")
    parser.add_argument('--test',default="False")
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


app = Flask(__name__)
CORS(app)
api = Api(app)

"""
初始化模型
"""


clue_bilism_crf_file = base_dir + "/checkpoints/clue/bilstm-crf/epoch_153.pth"
msra_bilism_crf_file = base_dir + "/checkpoints/msra/bilstm-crf/epoch_127.pth"
args = get_args()
print(args)
clue_data_set = DataSet("clue",base_dir = base_dir)
clue_data_set.word_to_id(args.min_freq)
msra_data_set = DataSet("msra",base_dir = base_dir)
msra_data_set.word_to_id(args.min_freq)
msra_data_set.extend_maps()
clue_data_set.extend_maps()
clue_model = BILSTM_Model(args, clue_data_set)
clue_model.model.load_state_dict(torch.load(clue_bilism_crf_file, map_location=args.device))
msra_model = BILSTM_Model(args, msra_data_set)
msra_model.model.load_state_dict(torch.load(msra_bilism_crf_file, map_location=args.device))


@app.route('/', methods=["GET"])
def index():
    return "try /cluener/<sentence> or /msraner/<sentence>."
class ChineseNERByCLUE(Resource):
    def get(self,sentence):
        tag_list = clue_ner(sentence)
        result = {"tag_list":tag_list,
                  "word_dict":get_ner_result(sentence,tag_list)}
        print(result)
        return result
    def post(self):
        print( request.json())
        return request.json()
class ChineseNERByMSRA(Resource):
    def get(self,sentence):
        print(sentence)
        tag_list = msra_ner(sentence)
        result = {"tag_list":tag_list,
                  "word_dict":get_ner_result(sentence,tag_list)}
        print(result)
        return result
def get_ner_result(sentence,tag_list):
    word_dict = {}
    word = ""
    tags  = ""
    lens = 0
    for i in range(len(sentence)):
        if tag_list[i] != "o":
            word +=sentence[i]
            tags =tag_list[i][2:]
            lens +=1
        else:
            if lens != 0 :
                word_dict[word] = tags
                lens = 0
                word = ""
                tags = ""
    return word_dict



def clue_ner(sentence):
    tag_list = clue_model.sentence_ner(sentence)
    tag_list = tag_list[0]
    return tag_list

def msra_ner(sentence):
    tag_list = msra_model.sentence_ner(sentence)
    tag_list = tag_list[0]
    return tag_list
api.add_resource(ChineseNERByCLUE, '/cluener/<sentence>')
api.add_resource(ChineseNERByMSRA, '/msraner/<sentence>')

if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=8010)
    app.run(debug=True)
