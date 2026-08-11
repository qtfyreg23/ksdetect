# exp_00_sanity_check

两项相互独立的检查，分开运行，依据项目计划 §4.7（"退出标准"）。

## 第一部分 —— 统计健全性检查（先运行这个，任何地方都能跑，不需要 GPU）

```bash
conda activate ksdetect
cd /root/autodl-tmp/ksdetect
python experiments/exp_00_sanity_check/run_stats_sanity_check.py
```

这一部分检查了什么，以及为什么每一项都很重要：

| 检查项 | 证明了什么 | 之后会在哪里用到 |
|---|---|---|
| (a) 强信号情形，置信区间下界 > 0.75 | `core.stats` 能正确检测出真实存在的信号 | 之后所有实验中的每一次 AUC 比较 |
| (b) 纯噪声情形，AUC 的置信区间覆盖 0.5 | 在没有真实信号时不存在系统性的向上偏差 | 排除大范围扫描中的假阳性 |
| (c) 经过筛选的特征（L1）与随机特征（random k）配对，Wilcoxon 检验 p<0.05 | 特征选择确实找到了超出随机水平的东西 | 项目计划要求的、在信任任何稀疏性结论之前必须做的"选择必要性"消融实验 |
| (d) 有泄漏版本 vs 正确版本，有泄漏版本的结果明显虚高 | 如果上一个项目中出现过的特征选择泄漏 bug（`docs/known_issues.md` #3）在这套流程里再次出现，能够被检测出来 | 回归保护——每次修改 `core/stats` 之后都运行一次这个检查 |
| (e) 词集重叠（word-set-overlap）标注的各种测试用例 | `core.labeling.is_correct` 的行为与文档描述的一致，而不是简单的子串匹配 | 之后每个实验中的每一个软/硬标签 |

**预期结果：** `results/summary.json` 中 `overall_PASS: true`，退出码
为 0。
如果失败，**不要**进入任何其他实验，先修复 `core/stats` 或
`core/labeling`，因为后续所有内容都依赖于它们（项目计划 §2.5，
"已经实现过的功能应先找到原实现并复用"——这里就是所有其他部分都会
调用的那个"原实现"）。

输出文件：`results/summary.json`、`results/fig_leakage_check.pdf`、
`results/fig_selected_vs_random.pdf`、`run_log_stats.txt`。

## 第二部分 —— 提取过程冒烟测试（仅限服务器，需要真实模型）

```bash
conda activate ksdetect
cd /root/autodl-tmp/ksdetect
python experiments/exp_00_sanity_check/run_extraction_smoke_test.py
```

在 `config.yaml` 中 `smoke_test_prompts` 里的 3 条 prompt 上运行，
检查以下内容：
  - 模型能从 `MODEL_PATH` 加载，且不触发任何网络请求；
  - hook 在全部四种模块类型（embedding、attention、ffn_module、
    ffn_neuron、residual）上都能正常触发，并产出符合预期形状的张量；
  - `extract_batch` 的（"last"）池化方式为每个 prompt 在每一层/每个
    模块都恰好产出一个向量，且不含 NaN。

在进行任何大规模提取任务**之前**，先运行这一步（项目计划 §2.2，
"先在少量案例上验证实现和实际行为，确认正确后再扩大规模"）。如果这一步
失败，问题几乎肯定出在 `core/extraction/hooks.py` 对 transformers
模块结构的假设上（见该文件的 docstring）——请先检查已安装的
`transformers` 版本是否与 `environment.yml` 中固定的版本一致。

**运行完这一步之后，请把以下内容发给我：**
1. 完整的控制台输出（或 `run_log_extraction.txt`）
2. 打印的是 `EXTRACTION SMOKE TEST: PASS` 还是 `FAIL`
3. 如果是 FAIL：完整的报错堆栈（traceback）

我会先确认这些形状/数值看起来没问题，你再进入第一阶段（大范围扫描）。
