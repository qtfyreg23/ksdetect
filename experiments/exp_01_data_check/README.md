# exp_01_data_check

在写完整的阶段1粗扫抽取代码之前,先验证`core/data`里的5个数据集加载器是否
和真实数据的字段结构一致。分两步,第一步要联网,第二步之后完全离线。

## 第一步:下载数据集到本地(需要联网,只需要跑一次)

```bash
conda activate ksdetect
cd /root/autodl-tmp/ksdetect
python scripts/download_datasets.py
```

会把5个数据集的完整原始数据(所有split)存到`data/raw_datasets/{name}/`
下(Arrow格式,用`datasets`库的`save_to_disk`)。这一步之后,`core/data`里
所有的加载器都只读本地磁盘,不再有任何网络请求——避免每次跑实验都依赖网络
状态,也避免重复下载。

如果只想重新下载某一个(比如某个失败了):
```bash
python scripts/download_datasets.py --only coqa
```

已经下载过的会自动跳过,除非加`--force`。

**这一步唯一会用到`trust_remote_code=True`的地方是CoQA**(它是较老的脚本
式HF加载器),而且只在下载时执行一次,下载完之后的本地Arrow数据不需要这
个参数就能读。如果这一步报错或有非预期的提示,发给我看一下。

## 第二步:验证字段结构(不需要联网)

```bash
python experiments/exp_01_data_check/run.py
```

## 检查内容

- 5个数据集(TruthfulQA/TriviaQA/CoQA/MedQuad/GSM8K)各加载10条,校验统一
  schema(example_id/question/references/task_type/dataset)每个字段都非空
- 打印每个数据集的一条样例,方便你我一起核对question/references的内容是否
  符合预期(尤其CoQA是拼接了story+问题,MedQuad有一部分answer为null已过滤)
- GSM8K额外做了一个自匹配检查:把真实的reference答案文本喂给
  `gsm8k_is_correct`当"预测",应该匹配成功——这验证的是数字解析逻辑本身,
  不是模型能力

## 预期结果

`results/summary.json`里`overall_PASS: true`,退出码0。

**跑完之后请把完整控制台输出(两步都要,包括下载步骤的日志)发给我**,我
需要确认:
1. 每个数据集的样例内容(question/references)看起来是否合理
2. 下载步骤有没有报错或异常提示,尤其CoQA
3. 各数据集实际能加载到的总量——TruthfulQA全量只有817条,肯定到不了
   2000条的粗扫目标,届时会直接用全量,我们到时候一起确认这个不影响整体
   设计

确认没问题后,我再写完整的阶段1粗扫抽取流水线(数据加载→模型生成/pregen
抽取→打标签→存activation shard→跑`core.stats`出结果)。
