# BertForClassification
Bert fine-tuning for Sentiment Analysis 基于Bert的情感分类模型微调

### 简介

基于Transformers库中BERT模型的情感三分类任务（消极、中性、积极），支持ONNX格式导出

训练前先安装必要环境pip install -r ，并更新配置文件 `config.yaml` 

Dataset可替换为自己的数据：分文件保存，lables与文件顺序对应

> 配置文件中：

> `describe` 参数指定了模型名称

> `restore_train` 参数控制训练断点恢复

> `save_all_epoch` 参数为True时，模型将在每次评估后保存一次模型。

> 仅当上次训练save_all_epoch设为True时，本次才能启用断点恢复。若断点恢复被启用，模型将默认依据describe加载最新的模型，否则将依据describe迭代新的模型版本

### 训练

DeepSpeed多卡GPU加速训练：

```shell
sh train.sh
```

普通训练：

```shell
python launch.py
```



### 推理

`test.py` 中包含模型的测试方法，不过在这之前你需要先将模型导出为ONNX



### 导出ONNX

运行 `gen_onnx.sh` 脚本，生成的ONNX模型将保存在配置文件中指定的文件夹

若配置文件中 `onnx.describe` 未指定（指定时需携带版本编号），则默认指向 `describe` 的最新模型迭代

当指定模型版本时（例：bert_v1），模型将加载指定版本

```
sh gen_onnx.sh
```

> PS：Model文件夹下的模型layer是之前写着玩的，可以直接删除