#! /bin/bash
"""
数据处理
"""
#处理clue数据集数据
python3 data_processing_clue.py
#处理msra数据集数据
python3 data_processing_msra.py


"""
训练模型
"""
#
#训练HMM
python3 main.py --algorithm="hmm" --data-set="clue" --min-freq=0
python3 main.py --algorithm="hmm" --data-set="msra" --min-freq=0
#训练CRF
python3 main.py --algorithm="crf" --data-set="clue" --min-freq=0
python3 main.py --algorithm="crf" --data-set="msra" --min-freq=0

#训练Bilstm-crf
python3 main.py --algorithm="bilstm-crf" --data-set="clue" --min-freq=10  --embedding-size=128  --hidden-size=256 --num-layers=2 --use-norm --use-dropout  --dropout=0.7 --epoch=100
python3 main.py --algorithm="bilstm-crf" --data-set="msra" --min-freq=10  --embedding-size=128  --hidden-size=256 --num-layers=2 --use-norm --use-dropout  --dropout=0.6 --epoch=200
#训练Bert-bilstm-crf
python3 main.py --algorithm="bilstm-crf" --data-set="clue" --min-freq=10  --use-bert --embedding-size=768  --hidden-size=200 --num-layers=2 --bert-model-dir="/bert/bert-base-chinese" --use-norm --use-dropout  --dropout=0.4 --epoch=1000 --lr=0.001 --crf-lr=0.005 --bert-lr=0.001
python3 main.py --algorithm="bilstm-crf" --data-set="msra" --min-freq=10  --use-bert --embedding-size=768  --hidden-size=200 --num-layers=2 --bert-model-dir="/bert/bert-base-chinese" --use-norm --use-dropout  --dropout=0.4 --epoch=1000 --lr=0.001 --crf-lr=0.005 --bert-lr=0.001

#训练Transformer-crf
python3 main.py --algorithm="transformer-crf" --data-set="clue" --min-freq=10  --d-model=128 --num-blocks=2  --num-heads=4  --feedforward-dim=512 --use-dropout  --dropout=0.1 --epoch=100
python3 main.py --algorithm="transformer-crf" --data-set="msra" --min-freq=10  --d-model=128 --num-blocks=2  --num-heads=4  --feedforward-dim=512 --use-dropout  --dropout=0.1 --epoch=100

#训练Bert-crf
python3 main.py --algorithm="bert-crf" --data-set="clue" --min-freq=10  --hidden-size=200 --bert-model-dir="/bert/bert-base-chinese" --use-dropout  --dropout=0.7 --epoch=300  --lr=5e-4 --crf-lr=5e-3 --bert-lr=1e-3
python3 main.py --algorithm="bert-crf" --data-set="msra" --min-freq=10  --hidden-size=400 --bert-model-dir="/bert/bert-base-chinese" --use-dropout  --dropout=0.6 --epoch=1000  --lr=1e-3 --crf-lr=1e-2  --bert-lr=1e-3

"""
测试模型
"""
#测试HMM
python3 main.py --algorithm="hmm" --data-set="clue" --min-freq=0 --test --test-model-path="/checkpoints/clue/hmm/hmm.pkl"
python3 main.py --algorithm="hmm" --data-set="msra" --min-freq=0 --test --test-model-path="/checkpoints/msra/hmm/hmm.pkl"
#测试CRF
python3 main.py --algorithm="crf" --data-set="clue" --min-freq=0 --test --test-model-path="/checkpoints/clue/crf/crf.pkl"
python3 main.py --algorithm="crf" --data-set="msra" --min-freq=0 --test --test-model-path="/checkpoints/msra/crf/crf.pkl"

#测试Bilstm-crf
python3 main.py --algorithm="bilstm-crf" --data-set="clue" --min-freq=10 --test --test-model-path="/checkpoints/clue/bilstm-crf/epoch_153.pth"  --embedding-size=128  --hidden-size=256 --num-layers=2 --use-norm
python3 main.py --algorithm="bilstm-crf" --data-set="msra" --min-freq=10 --test --test-model-path="/checkpoints/msra/bilstm-crf/epoch_127.pth"  --embedding-size=128  --hidden-size=256 --num-layers=2 --use-norm
#测试Bert-bilstm-crf
python3 main.py --algorithm="bilstm-crf" --data-set="clue" --min-freq=10 --test --test-model-path="/checkpoints/clue/bilstm-crf-bert/epoch_548.pth" --use-bert --embedding-size=768  --hidden-size=200 --num-layers=2 --bert-model-dir="/bert/bert-base-chinese" --use-norm
python3 main.py --algorithm="bilstm-crf" --data-set="msra" --min-freq=10 --test --test-model-path="/checkpoints/msra/bilstm-crf-bert/epoch_634.pth" --use-bert --embedding-size=768  --hidden-size=200 --num-layers=2 --bert-model-dir="/bert/bert-base-chinese" --use-norm


#测试Transformer-crf
python3 main.py --algorithm="transformer-crf" --data-set="clue" --min-freq=10 --test --test-model-path="/checkpoints/clue/transformer-crf/epoch_462.pth" --d-model=128 --num-blocks=1  --num-heads=2  --feedforward-dim=256
python3 main.py --algorithm="transformer-crf" --data-set="msra" --min-freq=10 --test --test-model-path="/checkpoints/msra/transformer-crf/epoch_1769.pth" --d-model=256 --num-blocks=4  --num-heads=8  --feedforward-dim=256

#测试Bert-crf
python3 main.py --algorithm="bert-crf" --data-set="clue" --min-freq=10 --test  --hidden-size=200 --test-model-path="/checkpoints/clue/bert-crf/epoch_475.pth" --batch-size=32
python3 main.py --algorithm="bert-crf" --data-set="msra" --min-freq=10 --test  --hidden-size=400 --test-model-path="/checkpoints/msra/bert-crf/epoch_235.pth" --batch-size=32


"""
#使用模型对单个句子进行NER任务
"""

#HMM   clue数据集的模型
python3 main.py --algorithm="hmm" --data-set="clue" --min-freq=0 --sentence-ner --test-model-path="/checkpoints/clue/hmm/hmm.pkl"
#HMM   msra数据集的模型
python3 main.py --algorithm="hmm" --data-set="msra" --min-freq=0 --sentence-ner --test-model-path="/checkpoints/msra/hmm/hmm.pkl"



#CRF  clue数据集的模型
python3 main.py --algorithm="crf" --data-set="clue" --min-freq=0 --sentence-ner --test-model-path="/checkpoints/clue/crf/crf.pkl"
#CRF   msra数据集的模型
python3 main.py --algorithm="crf" --data-set="msra" --min-freq=0 --sentence-ner --test-model-path="/checkpoints/msra/crf/crf.pkl"



#BiLSTM-crf  clue数据集的模型
python3 main.py --algorithm="bilstm-crf" --data-set="clue" --min-freq=10 --sentence-ner --test-model-path="/checkpoints/clue/bilstm-crf/epoch_153.pth"  --embedding-size=128  --hidden-size=256 --num-layers=2 --use-norm
#BiLSTM-crf  msra数据集的模型
python3 main.py --algorithm="bilstm-crf" --data-set="msra" --min-freq=10 --sentence-ner --test-model-path="/checkpoints/msra/bilstm-crf/epoch_127.pth"  --embedding-size=128  --hidden-size=256 --num-layers=2 --use-norm


#Bert-BiLSTM-crf  clue数据集的模型
python3 main.py --algorithm="bilstm-crf" --data-set="clue" --min-freq=10 --sentence-ner --test-model-path="/checkpoints/clue/bilstm-crf-bert/epoch_548.pth" --use-bert --embedding-size=768  --hidden-size=200 --num-layers=2 --bert-model-dir="/bert/bert-base-chinese" --use-norm
python3 main.py --algorithm="bilstm-crf" --data-set="msra" --min-freq=10 --sentence-ner --test-model-path="/checkpoints/msra/bilstm-crf-bert/epoch_634.pth" --use-bert --embedding-size=768  --hidden-size=200 --num-layers=2 --bert-model-dir="/bert/bert-base-chinese" --use-norm


#Transformer-crf clue数据集的模型
python3 main.py --algorithm="transformer-crf" --data-set="clue" --min-freq=10 --sentence-ner --test-model-path="/checkpoints/clue/transformer-crf/epoch_462.pth" --d-model=128 --num-blocks=1  --num-heads=2  --feedforward-dim=256
#Transformer-crf  msra数据集的模型
python3 main.py --algorithm="transformer-crf" --data-set="msra" --min-freq=10 --sentence-ner --test-model-path="/checkpoints/msra/transformer-crf/epoch_1769.pth" --d-model=256 --num-blocks=4  --num-heads=8  --feedforward-dim=256



#Bert-CRF clue数据集的模型
python3 main.py --algorithm="bert-crf" --data-set="clue" --min-freq=10 --sentence-ner  --hidden-size=200 --test-model-path="/checkpoints/clue/bert-crf/epoch_475.pth"
#Bert-CRF msra数据集的模型
python3 main.py --algorithm="bert-crf" --data-set="msra" --min-freq=10 --sentence-ner  --hidden-size=400 --test-model-path="/checkpoints/msra/bert-crf/epoch_235.pth"










