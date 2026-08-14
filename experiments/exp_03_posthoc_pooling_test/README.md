# exp_03_posthoc_pooling_test

阶段3竞争性假设设计的落地:验证"TruthfulQA/MedQuad里posthoc反而比pregen弱"
这个现象,到底是pooling方式的测量artifact(解释A),还是真实的表征层面
效应(解释B)。

**不重新生成任何文本**——直接复用`exp_02_coarse_scan`已经存好的
`greedy_answer`,只重新抽一遍posthoc的激活(换成"只在答案token范围内做
mean pooling",而不是"最后一个token"),然后和已有的pregen、posthoc(last)
做三方配对比较。

## 判定规则(实验前定死,不能看着数据事后调整)

- posthoc_answermean的均值 ≥ pregen,或者配对检验对pregen不显著(p≥0.05)→ **倾向解释A**(pooling artifact,原来的"posthoc更弱"是测量问题)
- posthoc_answermean依然显著低于pregen,且相对posthoc_last也没有显著提升 → **倾向解释B**(真实的表征层面效应)
- posthoc_answermean依然显著低于pregen,但相对posthoc_last有显著提升 → **mixed**(两种效应都有部分贡献)

## 运行顺序

### 第0步:免费定性预检(不需要模型/GPU,先跑这个)

```bash
conda activate ksdetect
cd /root/autodl-tmp/ksdetect
python experiments/exp_03_posthoc_pooling_test/run_00_qualitative_check.py
```

看TruthfulQA/MedQuad的`greedy_answer`结尾词分布够不够单调。这一步不能单独
下结论(哪怕结尾很单调,也只是给解释A增加一点可信度,不是证据本身),但
几乎不花钱,先看一眼心里有个数。

### 第1步:重新抽取posthoc(answer_mean),只针对这两个数据集

```bash
python experiments/exp_03_posthoc_pooling_test/run_01_extract_answer_mean.py
```

支持中断续跑(用`.done`标记文件判断,和之前的机制一致)。这一步只需要
TruthfulQA(817条)+MedQuad(1000条)两个数据集,规模比阶段1粗扫小得多,
预计耗时应该明显短于MedQuad在粗扫时的量级——但同样建议先看看跑前几个batch
的日志确认没问题,不用像之前那样单独跑pilot(规模已经足够小)。

### 第2步:三方配对比较,出判定结果

```bash
python experiments/exp_03_posthoc_pooling_test/run_02_compare.py
```

纯CPU,读取第1步的新数据+已有的pregen/posthoc(last)数据,对每个
(数据集×模块×层)算三者的AUC、两组配对显著性检验,给出该cell的判定
(A_supported/B_supported/mixed)。**排除了`embedding`模块**——pregen阶段
embedding是已知的退化情况(固定chat模板后缀token导致向量恒定,AUC恒为
0.5),拿它做比较没有意义,不是遗漏。

## 跑完之后发给我

1. `results/qualitative_check_report.txt`(第0步)
2. `results/summary_extract.json` + `run_log_extract.txt`尾部(第1步,确认有没有failed_batches)
3. `results/discrimination_results.csv` + `results/verdict_summary.txt`(第2步,这是最终判定)

拿到判定结果后,我们再决定:如果多数cell判A,说明这次"发现"要撤回,重新
用answer_mean版本的posthoc数据看阶段1的全貌;如果多数判B,这才是真正进
入阶段4(机制验证)、值得往下挖的现象。

---

## 更新(2026-08-14):2数据集试跑的结论 + 推广到全部5数据集

TruthfulQA全部18个cell完成,16个判A、2个判mixed,`p_vs_last`几乎全部
<0.0001——**强烈支持解释A(pooling artifact)**,原来"posthoc<pregen是真
实masking效应"这个现象假设不成立,予以撤回(见`docs/decisions.md` D18)。

因为这个pooling问题是聊天模板机制本身导致的,和数据集内容无关,大概率系
统性影响了阶段1粗扫全部5个数据集的posthoc数据,不只是这两个。用同一套脚
本、全新的config推广:

```bash
conda activate ksdetect
cd /root/autodl-tmp/ksdetect

# 第1步(GPU,复用已有的greedy_answer,不重新生成)
python experiments/exp_03_posthoc_pooling_test/run_01_extract_answer_mean.py \
    --config experiments/exp_03_posthoc_pooling_test/config_full_rescan.yaml

# 第2步(纯CPU)
python experiments/exp_03_posthoc_pooling_test/run_02_compare.py \
    --config experiments/exp_03_posthoc_pooling_test/config_full_rescan.yaml
```

**在跑第2步之前,强烈建议把CPU无卡实例的内存从2GB调大一档**(比如4GB,
按0.1元/小时的量级,多花的钱可以忽略不计)——2数据集试跑时就在
`ffn_neuron`(14336维)那个cell上因为内存不够被系统杀掉过一次,这次要跑
5个数据集、180个cell,同样的问题大概率还会复现好几次。代码这边已经加了
两处缓解(每个cell算完立刻落盘续跑、显式清内存),但这些是缓解不是根治,
根治还是加内存最直接。

如果还是中途被杀,直接重新跑同一条命令就行,续跑机制会跳过已经算完的
cell。

跑完之后把`results_full_rescan/`下的`summary_extract.json`、
`discrimination_results.csv`、`verdict_summary.txt`发给我,我们根据全量
判定结果决定要不要把阶段1粗扫的posthoc数据整体订正(重新生成
`exp_02b_analysis`的热力图)。
