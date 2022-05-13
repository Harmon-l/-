# Homework 1
## CIFAR100数据集
在config.py中设置数据集储存位置，首次运行train.py文件时会自动下载训练集

## 模型训练
在config.py中设置数据增强方式。
有 None/cutmix/cutout/mixup 选择，分别对应不进行数据增强以及对应的数据增强方式
运行下方命令进行训练
```
python train.py --work-dir ./
```

## 模型结果
 增强方法  | Backbone  | Accuracy
 ---- | ----- | ------  
 None  | Resnet50 | 77.83 
 CutMix  | Resnet50 | 80.36  
 CutOut  | Resnet50 | 78.47  
 Mixup  | Resnet50 | 79.26  

## 模型储存地址
已训练完成的模型上传在百度网盘中
网盘链接(https://pan.baidu.com/s/1w6lx5SfxE8nMS6u4IvWwZQ)
提取码：cot5

