# exp_06_depth_correlation_recheck

用订正后的posthoc pooling,重新检查阶段1粗扫里"层深度依赖任务类型"这个
候选现象是否还站得住——之前那批数据里TruthfulQA/TriviaQA(posthoc)显示
"深度敏感",但当时用的是有问题的last-token pooling,不能排除这个"深度敏
感"本身就是pooling artifact在深浅层上表现不均匀造成的假象。

## 只需要给GSM8K补抽数据,其余复用已有结果

按平均答案长度判断front100窗口和整段平均是否有实质区别:MedQuad/GSM8K
(120+词)需要订正,TruthfulQA(~80词,exp_05已经抽过)、TriviaQA/CoQA
(20词左右,front100基本等于exp_03已有的整段平均)不需要新抽。具体见
`config.yaml`里的`posthoc_source_by_dataset`,每个数据集用哪份数据是明
写死的,不是脚本运行时猜的。

## 运行顺序

```bash
conda activate ksdetect
cd /root/autodl-tmp/ksdetect

# 第1步(GPU,只针对GSM8K,复用已有答案文本)
python experiments/exp_06_depth_correlation_recheck/run_01_extract_gsm8k_windows.py

# 第2步(纯CPU):重新算层深度相关性,和旧数据对比
python experiments/exp_06_depth_correlation_recheck/run_02_depth_correlation_analysis.py

# 第3步(纯CPU):浅层饱和数据集的表层捷径审计
python experiments/exp_06_depth_correlation_recheck/run_03_shallow_shortcut_audit.py
```

第2步会自动尝试读取`experiments/exp_02b_analysis/results/wide_scan_results.csv`
(阶段1粗扫的原始结果)做新旧对比——如果这个文件还在就会自动纳入对比,不
在也不会报错,只是没有新旧对比这一栏。

**第2步的输出请先看`corrected_depth_results.csv`(逐层逐模块的原始数
字),不要只看`depth_correlation_comparison.csv`里汇总的相关系数**——
`run_03_optimal_window_check`那次已经吃过亏,汇总数字会把层间的真实波动
抹掉,显得比实际干净。看完逐层数据、确认哪些数据集看起来还是"浅层饱和"
之后,再决定`config.yaml`里`shallow_audit_datasets`要不要调整,再跑第3
步。

## 跑完发我

`results/corrected_depth_results.csv`、`results/depth_correlation_comparison.csv`、
`results/shallow_shortcut_audit.json`,外加你自己看逐层数据后的第一印象
(哪些数据集的"两簇分组"看起来还在、哪些变了)——这个判断我们一起看,不
是我单方面从数字里下结论。
