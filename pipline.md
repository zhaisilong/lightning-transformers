# pl-transformers

## 测试 pl-transformers

```shell
python train.py task=nlp/translation dataset=nlp/translation/wmt16
```

## 从 csv 数据集到 json 数据集

preprocessing.ipynb


```shell
python train.py \
    task=nlp/translation \
    dataset=nlp/translation/wmt16 \
    dataset.cfg.train_file=train.json \
    dataset.cfg.valid_file=valid.json
```

## DataModule

