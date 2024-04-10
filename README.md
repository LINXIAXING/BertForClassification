# BertForClassification
Bert fine-tuning for Sentiment Analysis 基于Bert的情感分类模型微调

### 简介

基于Transformers库中BERT模型的情感三分类任务（消极、中性、积极），支持ONNX格式导出。

训练前先安装必要环境pip install -r ，并更新配置文件 `config.yaml` 。其中Dataset可替换为自己的数据：分文件保存，lables与文件顺序对应。

> PS：Model文件夹下的模型layer是之前写着玩的，可以直接删除
>

### 训练

DeepSpeed双卡加速训练：

```shell
accelerate-launch --num_processes 2 trainer.py
```

普通训练：

```shell
python trainer.py
```



### 推理

`test.py` 中包含模型的测试方法，不过在这之前你需要先将模型导出为ONNX



### 导出ONNX

运行 `utils/generate_onnx.py` 脚本，生成的ONNX模型将保存在utils下的文件夹中

```
python utils/generate_onnx.py
```

