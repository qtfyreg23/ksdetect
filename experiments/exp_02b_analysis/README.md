# exp_02b_analysis

读取`exp_02_coarse_scan`产出的activation shard和标签,对每个
(数据集×阶段×信号来源×层×标签类型×k值)组合跑`core.stats.repeated_cv_auc`,
产出一张完整的结果表(CSV)和几张热力图。**纯CPU计算,不需要GPU**,可以用
同一台机器跑,也可以把`data/activations/exp_02_coarse_scan`和
`experiments/exp_02_coarse_scan/results`这两个目录拷到别的机器上跑。

这一步**不下任何结论**,只产出数据和图,现象筛选(阶段2)是看完这些结果
之后单独讨论的事。

## 网格规模说明

5数据集 × 2阶段(pregen/posthoc) × 5信号来源 × 9层 × 2标签类型 × 3档k值
(32/256/full),每个格子还要跑`n_seeds=5 × n_splits=5=25`折交叉验证——组
合数不小,而且是纯CPU工作,所以和抽取阶段一样,**先跑pilot实测一个格子的
真实耗时**,不要直接上全量。

## 第一步:跑pilot(1数据集×1阶段×2信号来源×2层,含大小两种维度对比耗时)

```bash
conda activate ksdetect
cd /root/autodl-tmp/ksdetect
python experiments/exp_02b_analysis/run.py --config experiments/exp_02b_analysis/config_pilot.yaml
```

`run_log.txt`里每个格子都会打进度,重点看跑完之后的**总耗时和平均每个格
子耗时**,尤其对比一下`attention`(4096维)和`ffn_neuron`(14336维)在同
样k值下的速度差异,这样能推算出全量网格的真实耗时,而不是我瞎猜。

## 第二步:pilot看完耗时后再跑全量

```bash
python experiments/exp_02b_analysis/run.py --config experiments/exp_02b_analysis/config.yaml
```

支持中断续跑:每算完一个格子就往`results/wide_scan_results.jsonl`追加一
行,重启会自动跳过已经算过的格子。

## 关于"某些格子会报错"这件事,提前说明

如果日志里看到类似`FAILED: This solver needs samples of at least 2
classes`或`multiclass classification`这种报错,**不用慌,不是bug**——这
是某个格子的标签退化成单一类别导致的(比如某个数据集在`soft_binarized`阈
值下几乎全对或全错),代码会记录这个格子失败、跳过、继续跑下一个,不会拖
垮整个流程。这种情况本身也是有信息量的(说明这个数据集在这个标签类型下没
有区分度),会体现在最终CSV里缺失对应的行,不需要特殊处理。

## 跑完之后发给我

1. `results(_pilot)/run_log.txt`里的耗时/进度片段(用于推算全量规模是否
   要调整)
2. `results(_pilot)/wide_scan_results.csv`
3. 生成的热力图(`fig_heatmap_*.pdf`)——这几张图我们会一起看,决定阶段2
   现象筛选往哪个方向深挖,不是我单方面下结论
