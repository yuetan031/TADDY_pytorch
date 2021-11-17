# TADDY: Anomaly detection in dynamic graphs via transformer
This repo covers an reference implementation for the paper "[Anomaly detection in dynamic graphs via transformer](https://arxiv.org/pdf/2106.09876.pdf)" (TADDY).

![framework](framework.png)

Some codes are borrowed from [Graph-Bert](https://github.com/jwzhanggy/Graph-Bert) and [NetWalk](https://github.com/chengw07/NetWalk).

## Requirments
* Python==3.8
* PyTorch==1.7.1
* Transformers==3.5.1
* Scipy==1.5.2
* Numpy==1.19.2
* Networkx==2.5
* Scikit-learn==0.23.2

## Usage
### Step 0: Prepare Data
Example:
```
python 0_prepare_data.py --dataset uci
```

### Step 1: Train Model
Example:
```
python 1_train.py --dataset uci --anomaly_per 0.1
```

## Cite
If this code is helpful, please cite the original paper:
```
@ARTICLE{liu2021anomaly,
  author={Liu, Yixin and Pan, Shirui and Wang, Yu Guang and Xiong, Fei and Wang, Liang and Chen, Qingfeng and Lee, Vincent CS},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Anomaly Detection in Dynamic Graphs via Transformer}, 
  year={2021},
  doi={10.1109/TKDE.2021.3124061}}

```

Don't forget to press a "star" after using!
