# DeepFM-MovieLens
用DeepFM预测用户对电影的评分，使用MovieLens-1M数据集，并使用AUC和训练时间评估模型效果。

## 环境说明
```shell
conda create --name <env> --file requirements.txt
```
torchfm需要使用pip安装
```shell
pip install torchfm
```
环境中torchfm和numpy版本可能不匹配。需要手动安装numpy<1.20.0或修改torchfm库中的py文件，替换其中的np.int和np.long为np.int32和np.int64。
模型用到的数据集可从[https://files.grouplens.org/datasets/movielens/ml-1m.zip](https://files.grouplens.org/datasets/movielens/ml-1m.zip)下载，下载后将ratings.dat置于data/ml-1m/即可。