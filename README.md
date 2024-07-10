# Video Moment Retrieval in Practical Setting: A Dataset of Ranked Moments for Imprecise  Queries

The benchmark and dataset for the paper (Video Moment Retrieval in Practical Settings: A Dataset of Ranked Moments for Imprecise Queries)[https://arxiv.org/abs/2407.06597].

We recommend cloning the code, data, and feature files from the Hugging Face repository at [TVR-Ranking](https://huggingface.co/axgroup/TVR-Ranking).  This repository only includes the code for ReLoCLNet. You can download the other baseline models from [XML](https://huggingface.co/LiangRenjie/XML_RVMR) and [CONQUER](https://huggingface.co/LiangRenjie/CONQUER_RVMR).

![TVR_Ranking_overview](./figures/taskComparisonV.png)  




## Getting started
### 1. Install the requisites

The Python packages we used are listed as follows. Commonly, the most recent versions work well.  


```shell
conda create --name tvr_ranking python=3.11
conda activate tvr_ranking
pip install pytorch # 2.2.1+cu121
pip install tensorboard 
pip install h5py pandas tqdm easydict pyyaml
```

### 2. Download full dataset
For the full dataset, please go down from Hugging Face [TVR-Ranking](https://huggingface.co/axgroup/TVR-Ranking). \
The detailed introduction and raw annotations is available at [Dataset Introduction](data/TVR_Ranking/readme.md).


```
TVR_Ranking/
  -val.json                  
  -test.json                 
  -train_top01.json
  -train_top20.json
  -train_top40.json
  -video_corpus.json
```

### 3. Download features

For the query BERT features, you can download them from Hugging Face [TVR-Ranking](https://huggingface.co/axgroup/TVR-Ranking). \
For the video and subtitle features, please request them at [TVR](https://tvr.cs.unc.edu/).

```shell
tar -xf tvr_feature_release.tar.gz -C data/TVR_Ranking/feature
```

### 4. Training
```shell
# modify the data path first 
sh run_top20.sh
```
### 5. Inferring
The checkpoint can all be accessed from Hugging Face [TVR-Ranking](https://huggingface.co/axgroup/TVR-Ranking).
```shell
sh infer_top20.sh
```

## Experiment Results
### Baseline
The baseline performance of  $NDGC@40$ was shown as follows.
Top $N$ moments were comprised of a pseudo training set by the query-caption similarity.

| **Model**      | **Train Set Top N** | **IoU=0.3**  | |**IoU=0.5**  | |**IoU=0.7**  | |
|----------------|---------------------|--------------|--------------|--------------|--------------|--------------|--------------|
|                |                     | **Val** | **Test** | **Val** | **Test** | **Val** | **Test** |
| **XML**        | 1                   | 0.1077 | 0.1016 | 0.0775 | 0.0727 | 0.0273 | 0.0294 |
|                | 20                  | 0.2580 | 0.2512 | 0.1874 | 0.1853 | 0.0705 | 0.0753 |
|                | 40                  | 0.2408 | 0.2432 | 0.1740 | 0.1791 | 0.0666 | 0.0720 |
| **ReLoCLNet**  | 1                   | 0.1533 | 0.1489 | 0.1321 | 0.1304 | 0.0878 | 0.0869 |
|                | 20                  | 0.4039 | 0.4031 | 0.3656 | 0.3648 | 0.2542 | 0.2567 |
|                | 40                  | 0.4725 | 0.4735 | 0.4337 | 0.4337 | 0.3015 | 0.3079 |


###  ReLoCLNet Performance

| **Model**  | **Train Set Top N** | **IoU=0.3**  | |**IoU=0.5**  | |**IoU=0.7**  | |
|------------|---------------------|--------------|--------------|--------------|--------------|--------------|--------------|
|            |                     | **Val** | **Test** | **Val** | **Test** | **Val** | **Test** |
| **NDCG@10** |                     |              |              |              |              |              |              |
| ReLoCLNet  | 1                   | 0.1575 | 0.1525 | 0.1358 | 0.1349 | 0.0908 | 0.0916 |
| ReLoCLNet  | 20                  | 0.3751 | 0.3751 | 0.3407 | 0.3397 | 0.2316 | 0.2338 |
| ReLoCLNet  | 40                  | 0.4339 | 0.4353 | 0.3984 | 0.3986 | 0.2693 | 0.2807 |
| **NDCG@20** |                     |              |              |              |              |              |              |
| ReLoCLNet  | 1                   | 0.1504 | 0.1439 | 0.1303 | 0.1269 | 0.0866 | 0.0849 |
| ReLoCLNet  | 20                  | 0.3815 | 0.3792 | 0.3462 | 0.3427 | 0.2381 | 0.2386 |
| ReLoCLNet  | 40                  | 0.4418 | 0.4439 | 0.4060 | 0.4059 | 0.2787 | 0.2877 |
| **NDCG@40** |                     |              |              |              |              |              |              |
| ReLoCLNet  | 1                   | 0.1533 | 0.1489 | 0.1321 | 0.1304 | 0.0878 | 0.0869 |
| ReLoCLNet  | 20                  | 0.4039 | 0.4031 | 0.3656 | 0.3648 | 0.2542 | 0.2567 |
| ReLoCLNet  | 40                  | 0.4725 | 0.4735 | 0.4337 | 0.4337 | 0.3015 | 0.3079 |








## Citation
If you feel this project helpful to your research, please cite our work.
```

```
