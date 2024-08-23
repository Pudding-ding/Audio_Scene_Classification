# Audio_Scene_Classification By Ding
Audio Scene Classification with Pytorch for a final exam in College, Oldenburg Germany

This project is part of the exam in audio technology. The dataset is taken from the DCASE2016 Challenge~(https://dcase.community/challenge2016/). There exist 15 classes for acoustical scenes. The Python script is abel to recocognize pretty good scores in the classification task. Afterwards, the system can change the signalprocessing of a compressor, for adapt the signalprocessing to the acoustical situasjon.

## 当前环境信息
1. 当前系统信息: windows10
2. python: anaconda python3.11.7
3. pytorch 版本: CPU
4. pycharm 版本: 社区版，2023.3.5

> 已修复问题：
> - 修改some_tests 文件中 30行的数据格式问题

## 当前项目结构
```
|-Models 训练后的数据集
|-.gitignore
|-all_data.csv 所有训练数据的路径说明
|-CNN.py cnn算法核心类
|-Config.py 初始化一些配置信息，
|-dcase2016_taslp.pdf 比赛原文pdf
|-Kompressor 
|-pre_processing 预处理和处理工具
|-README.md
|-some_tests.py 环境测试
|-Validation.py 调用 pre_precessing 验证模型
|-validation_data 验证的数据集
```

> Author: DingDing

原有仓库地址：https://github.com/NiggoNiggo/Audio_Scene_Classification