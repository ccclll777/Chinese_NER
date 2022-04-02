#! /bin/bash

#处理clue数据集数据
python3 data_processing_clue.py
#处理msra数据集数据
python3 data_processing_msra.py

#训练模型
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
python3 main.py --algorithm="bilstm-crf" --data-set="clue" --min-freq=10  --use-bert --embedding-size=768  --hidden-size=200 --num-layers=2 --bert-model-dir="/bert/bert-base-chinese" --use-norm --use-dropout  --dropout=0.4 --epoch=300 --lr=1e-3 --crf_lr=3e-3
python3 main.py --algorithm="bilstm-crf" --data-set="msra" --min-freq=10  --use-bert --embedding-size=768  --hidden-size=200 --num-layers=2 --bert-model-dir="/bert/bert-base-chinese" --use-norm --use-dropout  --dropout=0.4 --epoch=300 --lr=1e-4 --crf_lr=3e-3
#训练Transformer-crf
python3 main.py --algorithm="transformer-crf" --data-set="clue" --min-freq=10  --d-model=128 --num-blocks=2  --num-heads=4  --feedforward-dim=512 --use-dropout  --dropout=0.1 --epoch=100
python3 main.py --algorithm="transformer-crf" --data-set="msra" --min-freq=10  --d-model=128 --num-blocks=2  --num-heads=4  --feedforward-dim=512 --use-dropout  --dropout=0.1 --epoch=100
#训练Bert-crf
python3 main.py --algorithm="bert-crf" --data-set="clue" --min-freq=10  --hidden-size=200 --bert-model-dir="/bert/bert-base-chinese" --use-dropout  --dropout=0.1 --epoch=300  --lr=3e-4 --crf-lr=3e-2  --bert-lr=3e-5
python3 main.py --algorithm="bert-crf" --data-set="msra" --min-freq=10  --hidden-size=200 --bert-model-dir="/bert/bert-base-chinese" --use-dropout  --dropout=0.1 --epoch=100  --lr=3e-4 --crf-lr=3e-2  --bert-lr=3e-5




#测试HMM
python3 main.py --algorithm="hmm" --data-set="clue" --min-freq=0 --test --test-model-path="/checkpoints/clue/hmm/hmm.pkl"
python3 main.py --algorithm="hmm" --data-set="msra" --min-freq=0 --test --test-model-path="/checkpoints/msra/hmm/hmm.pkl"
#测试CRF
python3 main.py --algorithm="crf" --data-set="clue" --min-freq=0 --test --test-model-path="/checkpoints/clue/crf/crf.pkl"
python3 main.py --algorithm="crf" --data-set="msra" --min-freq=0 --test --test-model-path="/checkpoints/msra/crf/crf.pkl"

#测试Bilstm-crf
python3 main.py --algorithm="bilstm-crf" --data-set="clue" --min-freq=10 --test --test-model-path="/checkpoints/clue/bilstm-crf/epoch_153.pth"  --embedding-size=128  --hidden-size=256 --num-layers=2 --use-norm --use-dropout   --dropout=0.7
python3 main.py --algorithm="bilstm-crf" --data-set="msra" --min-freq=10 --test --test-model-path="/checkpoints/msra/bilstm-crf/epoch_127.pth"  --embedding-size=128  --hidden-size=256 --num-layers=2 --use-norm --use-dropout  --dropout=0.6
#测试Bert-bilstm-crf
python3 main.py --algorithm="bilstm-crf" --data-set="clue" --min-freq=10 --test --test-model-path="/checkpoints/clue/bilstm-crf-bert/epoch_164.pth" --use-bert --embedding-size=768  --hidden-size=200 --num-layers=2 --bert-model-dir="/bert/bert-base-chinese" --use-norm --use-dropout  --dropout=0.4
python3 main.py --algorithm="bilstm-crf" --data-set="msra" --min-freq=10 --test --test-model-path="/checkpoints/clue/bilstm-crf-bert/epoch_10.pth" --use-bert --embedding-size=768  --hidden-size=200 --num-layers=2 --bert-model-dir="/bert/bert-base-chinese" --use-norm --use-dropout  --dropout=0.4


#测试Transformer-crf
python3 main.py --algorithm="transformer-crf" --data-set="clue" --min-freq=10 --test --test-model-path="/checkpoints/clue/transformer-crf/epoch_462.pth" --d-model=128 --num-blocks=1  --num-heads=2  --feedforward-dim=256 --use-dropout  --dropout=0.4
python3 main.py --algorithm="transformer-crf" --data-set="msra" --min-freq=10 --test --test-model-path="/checkpoints/clue/transformer-crf/epoch_10.pth" --d-model=128 --num-blocks=2  --num-heads=4  --feedforward-dim=512 --use-dropout  --dropout=0.1
