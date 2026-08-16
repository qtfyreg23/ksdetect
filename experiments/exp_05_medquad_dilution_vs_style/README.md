# exp_04_cross_dataset_check + exp_05_medquad_dilution_vs_style

这两个实验对应"MedQuad现象值不值得深挖"这个问题的两步验证:先便宜地检查
普适性,再正式做H1(稀释)vs H2(内容/文体掩盖)的区分实验。

## exp_04:跨数据集普适性检验(先跑这个,几分钟,纯CPU,不需要新抽取)

```bash
conda activate ksdetect
cd /root/autodl-tmp/ksdetect
python experiments/exp_04_cross_dataset_check/run_check.py
```

检查:数据集里"正确答案"和"错误答案"在文体上越难区分(用长度分布的
Cohen's d做粗糙代理),posthoc相对pregen的信号损失是否越大——这是H2成立
的话应该在5个数据集之间呈现的模式,不只是MedQuad自己。**n=5做不了严谨统
计推断**,这一步只是方向性参考,决定要不要真金白银投入下面的区分实验。

发我`results/style_vs_gap.csv`和`summary.json`。

## exp_05:H1 vs H2区分实验(正式实验,需要GPU重新抽取)

不管exp_04的结果如何,这个区分实验都值得做(即便exp_04没看出跨数据集模
式,也不代表MedQuad内部的H1/H2判定不重要,只是会影响我们怎么描述这个现
象的普适性范围)。

**做法**:把posthoc的pooling窗口从"整个答案"收窄到"答案开头前K个token"
(K=20/50/100),对比这几档窗口和已有的pregen/posthoc_last/posthoc_full
(整个答案)。

- **H1(稀释)成立的信号**:AUC随窗口变大单调上升,front20明显低于full,
  说明信息集中在后面,平均整个答案确实把它冲淡了
- **H2(掩盖)成立的信号**:front20已经和full差不多(甚至更早的位置就已
  经掉分了),说明不是"平均了多少"的问题,是"模型一开口"这件事本身就
  把信号盖住了

判定规则写死在`run_02_compare.py`的`verdict()`函数里,已经用两种方向相
反的构造数据验证过(见下方"已完成的验证")。

同时纳入了TruthfulQA作为对照——它在exp_03里已经证实"posthoc弱"是纯
artifact(用整个答案pooling就完全解决了),如果这次front-K扫描在
TruthfulQA上也看不出任何异常(不管K多小,AUC都不掉),这是符合预期的对
照结果,不是遗漏。

### 运行顺序

```bash
# 第1步:GPU,复用已有答案文本,重新抽取front-K窗口的posthoc激活
python experiments/exp_05_medquad_dilution_vs_style/run_01_extract_front_windows.py

# 第2步:纯CPU,六方比较+判定
python experiments/exp_05_medquad_dilution_vs_style/run_02_compare.py
```

第2步同样内存敏感(六份激活同时载入,`ffn_neuron`那几层依然吃内存),已
经带了和`exp_03`一样的续跑+内存清理机制,建议内存别再是2GB那个档位了。

### 跑完发我

`results/summary_extract.json`(第1步)、`results/h1_h2_results.csv`、
`results/verdict_summary.txt`(第2步)。

---

## 已完成的验证(供参考,不需要你操作)

- 前K窗口的索引逻辑(左padding场景下"取答案开头而非结尾的前K个token"有
  没有偏移)用纯numpy单独验证过,结果正确
- `run_02_compare.py`的判定逻辑用两种方向相反的构造数据分别测试过:
  - H2一致场景(front20≈front50≈front100≈full,全部明显低于pregen)→
    正确判定为`H2_supported`
  - H1一致场景(front20明显低于full,随K增大单调回升接近pregen)→正确
    判定为`H1_supported`
- 续跑机制、config-hash防护均复测通过
