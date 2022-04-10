# Homework 1
## MNIST数据集
前往 http://yann.lecun.com/exdb/mnist/ 下载MNIST数据集，并将数据集路径在 train.py 中更新
## 模型训练
首先设置epoch=10，设置需要查找的超参数范围，可在train.py中修改。训练结束会返回最佳的参数设置；
重新设置预训练获取的超参数，设置epoch=100，最终模型测试准确率为0.9767
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
train.py文件默认使用 learnrate=0.05，hidden=256，正则化系数alpha=0.0001
可以修改训练文件中的超参数返回最佳超参数
