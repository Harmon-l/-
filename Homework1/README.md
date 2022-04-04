# Homework 1
## MNIST数据集
前往 http://yann.lecun.com/exdb/mnist/ 下载MNIST数据集，并将数据集路径在 train.py 中更新
## 模型训练
运行train.py文件，训练模型会储存在运行文件路径下
```
python train.py
```
## 模型测试
运行evaluate.py文件，测试结果会自动打印
```
python evaluate.py
```
## 超参数
train.py文件默认使用 learnrate=0.01，hidden=256，正则化系数alpha=0.0001
可以修改训练文件中的超参数返回最佳超参数
