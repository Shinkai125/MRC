# HuggingFace-Transformers + Tensoflow2.0 Keras 做机器阅读理解

## 出发点
网上的机器阅读理解代码要么太冗余，要么封装的过于严重，所以自己写了一个HuggingFace-Transformers + Tensoflow2.0 Keras 做机器阅读理解代码，还在改进中，欢迎提意见。
## 特性
- HuggingFace-Transformers + Tensoflow2.0 Keras做机器阅读理解的简洁版本
- 清晰易懂的代码结构
- 支持MRC的答案位置修正
- 支持TPU

## 数据集下载和模型下载链接
- 数据集：[CMRC2018](https://bj.bcebos.com/paddlehub-dataset/cmrc2018.tar.gz)
- 模型：[RoBERTa-wwm-ext, Chinese](https://github.com/ymcui/Chinese-BERT-wwm)

## 准确率报告

|      模型       |  数据集  |         参数         | Eval EM | Eval F1 |
| :-------------: | :------: | :------------------: | :-----: | :-----: |
| RoBERTa-wwm-ext | CMRC2018 | TPU，Batch_Size = 32 | 79.559  | 59.211  |
|                 |          |                      |         |         |
|                 |          |                      |         |         |

## TODO
- 改进模型结构，提高正确率 
- 改进数据处理方式，提高正确率

## 欢迎提供改进意见
QQ群：957229713

## 参考和感谢
- CMRC2018
- HuggingFace-Transformers
- Keras