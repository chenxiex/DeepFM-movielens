# DeepFM-MovieLens
用DeepFM预测用户对电影的评分，使用MovieLens-1M数据集。

## 环境
```shell
conda create --name <env> --file requirements.txt
```
环境中torchfm和numpy版本可能不匹配。需要手动安装numpy<1.20.0或修改torchfm库中的py文件，替换其中的np.int和np.long为np.int32和np.int64。