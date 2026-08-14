# exp_02_coarse_scan

阶段1(宽扫描)第一轮粗扫的抽取流水线。这一步**只产出activation shard和
标签**,不做统计分析——统计/画图是单独一个脚本(`exp_02b`,还没写,等这一
步跑完确认数据没问题后再写),这样统计代码有bug不需要重新跑一遍GPU抽取。

## 运行前必读

- 每条样本要做:1次贪婪生成(硬标签+posthoc文本来源)+ 10次温度采样(软
  标签)+ 2次前向抽取(pregen + posthoc,9个粗采样层 × 5种信号来源)。计算
  量不小,务必先跑pilot。
- **posthoc的pooling这一轮粗扫简化成了和pregen一样的"last token"**,没有
  单独实现"只在生成的答案片段上做mean pooling"——如果后面发现posthoc信号
  很弱,这是第一个要复查的简化点(见`docs/decisions.md` D11)。
- 支持中断续跑:每个batch处理完,activation shard和标签shard都落盘后才算
  "完成",标签shard文件是否存在就是续跑时判断"这个batch要不要跳过"的依
  据。中途Ctrl+C或者掉线了,直接重新跑同一个命令就会接着跑,不会重复。
- **配置和输出目录强绑定**:如果你改了`config.yaml`里任何字段又想复用同
  一个`output_dir`续跑,脚本会直接报错拒绝启动,不会静默地把新旧设置的结
  果混在一起。想跑不同配置,换一个新的`output_dir`/`activation_dir`。

## 第一步:先跑pilot(8条/数据集,验证流程+观察显存)

```bash
conda activate ksdetect
cd /root/autodl-tmp/ksdetect
python experiments/exp_02_coarse_scan/run.py --config experiments/exp_02_coarse_scan/config_pilot.yaml
```

**跑之前开一个新终端,先跑`watch -n 1 nvidia-smi`盯着显存**,尤其是软标
签采样那一步(等效batch是`batch_size × n_samples_for_soft_label` =
4×10=40条序列同时生成),这是全流程里最吃显存的一步。如果这里OOM了,先调
小`config_pilot.yaml`里的`batch_size`(不要调`n_samples_for_soft_label`,
那是标签质量参数不是显存参数,调了要单独说明),不需要改代码。

## 第二步:pilot通过后,跑正式的粗扫

```bash
python experiments/exp_02_coarse_scan/run.py --config experiments/exp_02_coarse_scan/config.yaml
```

这一步预计耗时会长(具体多久要看pilot跑完后的实际速度换算,`run_log.txt`
里每5个batch会打一次进度,包含预计剩余时间,不用凭感觉猜)。可以放后台跑
(`nohup ... &`或者tmux/screen),中断了直接重跑同一条命令续上。

## 跑完之后发给我

无论pilot还是正式跑完,请提供:

1. **`results(_pilot)/summary.json`的完整内容**(每个数据集的
   completed/errored/failed_batches统计)
2. **`run_log.txt`里的进度日志片段**(尤其是速度rate和ETA那几行,让我判断
   有没有必要调整batch_size或者拆分成分批跑)
3. **如果有`failed_batches`,把对应的ERROR日志(带完整traceback)一起发
   过来**——不要先自己删了重跑,先让我看一眼报错原因
4. **随便挑一条`results/labels/{dataset}/shard_00000.jsonl`里的记录**发
   给我看看,确认question/greedy_answer/hard_label/soft_label的内容看起
   来合理(尤其GSM8K的greedy_answer是否真的在做推理、有没有正确带出
   "#### 数字"这种格式,这样后面数值匹配才准)

确认pilot没问题后再跑正式版,不要跳过pilot直接跑全量——2000×5个数据集的
规模一旦流程有问题,重跑成本很高。
