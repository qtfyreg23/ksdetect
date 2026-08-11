# 已知问题记录（Known Issues Log）

只追加的 bug、坑点及其修复方法记录。按照项目计划 §2.4（"每次犯过的
错误都应记录下来"），在排查看起来眼熟的问题之前先查这个文件，并且
每当修复了一个不那么小的 bug 时，就新增一条编号条目——即使事后看起来
"显而易见"也一样。

每条记录的格式：发生了什么、是怎么发现的、修复方法，以及现在是哪个/
哪些文件承载了这个修复，从而避免它悄悄地再次出现。

---

### #1 —— `datasets` 库的破坏性变更导致 `hotpot_qa` 加载失败
**继承自上一版项目**（记录在这里作为背景信息，尚未在当前代码库
`ksdetect` 中重新验证——第一次真正在 `ksdetect` 中加载 `hotpot_qa`
时需要重新验证）。
`datasets>=4.0.0` 引入了一处改动，导致之前使用的 `hotpot_qa` 加载
脚本失效。修复方法：`environment.yml` 明确固定 `datasets==2.21.0`，
而不是使用 `datasets>=X`。
**记录于：** `environment.yml`。

### #2 —— 单一随机种子的交叉验证会得出噪声较大、不稳定的结论
**继承自上一版项目。** 单次 5 折交叉验证只能得到 5 个 AUC 值；在 k
较小/样本量较小的情况下，"哪个 k 值最先越过某个阈值"实际上取决于用了
哪个 fold 划分的随机种子，而不是取决于底层信号本身。修复方法：始终
将多个随机种子下的 fold 级 AUC 汇总起来（`core.stats.repeated_cv_auc`
/ `matched_repeated_cv`），并且用置信区间的**下界**、而不是点估计值，
来判断是否越过阈值（`core.stats.ci_lower_bound_exceeds`）。
**记录于：** `core/stats/multiseed.py`、`core/stats/ci.py`。

### #3 —— 特征选择在交叉验证划分之前就在全量数据集上进行（数据泄漏）
**继承自上一版项目。** 早期版本用**全部**样本（包括后来落入测试
fold 的样本）来挑选"top-k"神经元，这导致了不合理的高 AUC（k=512 时
AUC 达到 0.99+），而一旦把特征选择正确地限定在每个 fold 的训练部分内
进行，这个虚高的结果就消失了。修复方法：`core.stats.nested_cv.
select_features()` 只接收训练 fold 的数据；一个故意保留泄漏漏洞的
版本（`cross_validated_auc_LEAKY_FOR_TESTING_ONLY`）仅作为回归测试
的夹具（fixture）保留在 `experiments/exp_00_sanity_check` 中，绝不会
在真正的实验代码里被调用。
**记录于：** `core/stats/nested_cv.py`；在
`experiments/exp_00_sanity_check/run_stats_sanity_check.py` 的 (d)
项检查中做了回归测试。

### #4 —— AutoDL 上的 HuggingFace 接口/网络配置问题
**继承自上一版项目。** 在 AutoDL 实例上通过 HF Hub API 加载模型时
遇到了接口/网络配置方面的问题。修复方法：始终从本地路径
（`/autodl-fs/data/Llama-3.1-8B-Instruct`）加载，并设置
`local_files_only=True`，绝不回退到从 Hub 下载。
**记录于：** `core/extraction/model_loader.py`。

### #5 —— 简单的子串匹配会导致正确答案的漏判或误判
**继承自上一版项目。** 严格的 `reference in prediction`（子串包含）
匹配方式，既会漏掉表述方式不同的正确答案，又会错误匹配到无关的短
reference。修复方法：在 `core.labeling.correctness.is_correct` 中
改用词集重叠（word-set overlap）匹配，并明确记录了不做的事项（不做
数字归一化、不做同义词消解），以避免这部分匹配逻辑的范围在不知不觉中
持续扩大。
**记录于：** `core/labeling/correctness.py`。

---

<!-- 新条目请添加在这一行下面。使用相同的格式：发生了什么、如何发现、
修复方法，以及哪个文件承载了这个修复。 -->
