# exp_07_medquad_shallow_signal_probe

承接 `docs/decisions.md` D26。exp_06 确认了 MedQuad 的浅层信号
(~0.83-0.87 AUC，layer0就有，corr(layer,AUC)只有0.02-0.18) 不是文本长度
捷径(exp_06/run_03已排除)。这一步区分两个互斥解释：

- **H_A**：这个信号本质是"粗粒度问题类型/模板分桶"能力，不是对具体疾病的
  细粒度判断。
- **H_C**：信号比"问题属于哪个模板桶"更细，粗粒度分桶特征解释不了。

判定不由脚本自动下（按D23的教训），你跑完看`shortcut_probe_summary.json`
里几个AUC读数，对照D26记录的真实信号范围(0.83-0.87)人工判断。

## 运行(纯CPU,不需要GPU/模型,几分钟内完成)

```bash
conda activate ksdetect
cd /root/autodl-tmp/ksdetect
python experiments/exp_07_medquad_shallow_signal_probe/run_01_template_shortcut_probe.py
```

## 一个已知的不确定性,跑完请特别看日志里这一段

脚本会尝试从`core.data.load_dataset_examples("medquad", ...)`里找原始
`question_type`字段(D8提到`lavita/MedQuAD`有这一列)，如果这个字段没有被
`core/data/loaders.py::load_medquad`透传下来（unified schema只保证
example_id/question/references/task_type/dataset这5个key），脚本会打印
清楚的warning、跳过这个特征、不会崩，用关键词模板桶(`keyword_buckets`)
作为不依赖这个字段的替代H_A操作化方式继续跑完。

**请把日志里`question_type: NOT AVAILABLE`还是找到了多少个example_id的
那几行发给我**——如果这个字段确实没保留，值得讨论要不要给`load_medquad`
补一个小patch把它带出来(不会破坏exp_01的REQUIRED_KEYS检查，那个检查只
查缺失key，不查多余key)，但这需要你先把`core/data/loaders.py`的真实内容
发我，我才能写精确的patch，而不是凭空猜一个函数签名。

## 跑完发我

`results/shortcut_probe_summary.json`、`results/run_log_shortcut_probe.txt`，
以及你看完之后第一印象是偏向H_A还是H_C（或者两者都不干净，比如某个具体
keyword bucket的AUC意外地高/低）——这个判断我们一起看。