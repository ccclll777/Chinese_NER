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
python3 main.py --algorithm="bilstm-crf" --data-set="clue" --min-freq=10  --embedding-size=128  --hidden-size=256 --num-layers=2 --use-norm --use-dropout  --dropout=0.1 --epoch=100
python3 main.py --algorithm="bilstm-crf" --data-set="msra" --min-freq=10  --embedding-size=128  --hidden-size=256 --num-layers=2 --use-norm --use-dropout  --dropout=0.1 --epoch=100
#训练Bert-bilstm-crf
python3 main.py --algorithm="bilstm-crf" --data-set="clue" --min-freq=10  --use-bert --embedding-size=768  --hidden-size=384 --num-layers=3 --bert-model-dir="/bert/768" --use-norm --use-dropout  --dropout=0.1 --epoch=100
python3 main.py --algorithm="bilstm-crf" --data-set="msra" --min-freq=10  --use-bert --embedding-size=768  --hidden-size=384 --num-layers=3 --bert-model-dir="/bert/768" --use-norm --use-dropout  --dropout=0.1 --epoch=100

python3 main.py --algorithm="bilstm-crf" --data-set="clue" --min-freq=10  --use-bert --embedding-size=1024  --hidden-size=512 --num-layers=3 --bert-model-dir="/bert/1024" --use-norm --use-dropout  --dropout=0.1 --epoch=100
python3 main.py --algorithm="bilstm-crf" --data-set="msra" --min-freq=10  --use-bert --embedding-size=1024  --hidden-size=512 --num-layers=3 --bert-model-dir="/bert/1024" --use-norm --use-dropout  --dropout=0.1 --epoch=100

#训练Transformer-crf
python3 main.py --algorithm="transformer-crf" --data-set="clue" --min-freq=10  --d-model=128 --num-blocks=2  --num-heads=4  --feedforward-dim=512 --use-dropout  --dropout=0.1 --epoch=100
python3 main.py --algorithm="transformer-crf" --data-set="msra" --min-freq=10  --d-model=128 --num-blocks=2  --num-heads=4  --feedforward-dim=512 --use-dropout  --dropout=0.1 --epoch=100




#测试HMM
python3 main.py --algorithm="hmm" --data-set="clue" --min-freq=0 --test --test-model-path="/checkpoints/clue/hmm/hmm.pkl"
python3 main.py --algorithm="hmm" --data-set="msra" --min-freq=0 --test --test-model-path="/checkpoints/msra/hmm/hmm.pkl"
#测试CRF
python3 main.py --algorithm="crf" --data-set="clue" --min-freq=0 --test --test-model-path="/checkpoints/clue/crf/crf.pkl"
python3 main.py --algorithm="crf" --data-set="msra" --min-freq=0 --test --test-model-path="/checkpoints/msra/crf/crf.pkl"

#测试Bilstm-crf
python3 main.py --algorithm="bilstm-crf" --data-set="clue" --min-freq=10 --test --test-model-path="/checkpoints/clue/bilstm-crf/epoch_10.pth"  --embedding-size=128  --hidden-size=256 --num-layers=2 --use-norm --use-dropout  --dropout=0.1
python3 main.py --algorithm="bilstm-crf" --data-set="msra" --min-freq=10 --test --test-model-path="/checkpoints/msra/bilstm-crf/epoch_10.pth"  --embedding-size=128  --hidden-size=256 --num-layers=2 --use-norm --use-dropout  --dropout=0.1
#测试Bert-bilstm-crf
python3 main.py --algorithm="bilstm-crf" --data-set="clue" --min-freq=10 --test --test-model-path="/checkpoints/clue/bilstm-crf-bert/epoch_10.pth" --use-bert --embedding-size=768  --hidden-size=384 --num-layers=3 --bert-model-dir="/bert/768" --use-dropout  --dropout=0.1
python3 main.py --algorithm="bilstm-crf" --data-set="msra" --min-freq=10 --test --test-model-path="/checkpoints/clue/bilstm-crf-bert/epoch_10.pth" --use-bert --embedding-size=768  --hidden-size=384 --num-layers=3 --bert-model-dir="/bert/768" --use-dropout  --dropout=0.1


#测试Transformer-crf
python3 main.py --algorithm="transformer-crf" --data-set="clue" --min-freq=10 --test --test-model-path="/checkpoints/clue/transformer-crf/epoch_10.pth" --d-model=128 --num-blocks=2  --num-heads=4  --feedforward-dim=512 --use-dropout  --dropout=0.1
python3 main.py --algorithm="transformer-crf" --data-set="msra" --min-freq=10 --test --test-model-path="/checkpoints/clue/transformer-crf/epoch_10.pth" --d-model=128 --num-blocks=2  --num-heads=4  --feedforward-dim=512 --use-dropout  --dropout=0.1
