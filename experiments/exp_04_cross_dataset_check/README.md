# exp_04_cross_dataset_check

见 `../exp_05_medquad_dilution_vs_style/README.md` 的完整说明——这两个实
验是同一条调查线的前后两步,写在一起方便对照。这里只重复运行命令:

```bash
conda activate ksdetect
cd /root/autodl-tmp/ksdetect
python experiments/exp_04_cross_dataset_check/run_check.py
```

不需要GPU,几分钟内跑完,只读已有的标签数据和`exp_03`的
`discrimination_results.csv`,不产生新的抽取任务。
