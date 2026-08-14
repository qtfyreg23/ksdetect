# Known Issues Log

Append-only log of bugs, gotchas, and their fixes. Per project plan §2.4
("每次犯过的错误都应记录下来"), check this file before debugging something
that looks familiar, and add a new numbered entry whenever you fix a
non-trivial bug — even if it seems "obvious" in hindsight.

Format per entry: what happened, how it was found, the fix, and which
file(s) now encode the fix so it can't silently regress.

---

### #1 — `datasets` library breaking change broke `hotpot_qa` loading
**Inherited from the earlier project iteration** (recorded here as
background, not yet re-verified against this codebase — re-verify the first
time `hotpot_qa` is actually loaded in `ksdetect`).
`datasets>=4.0.0` introduced a change that broke the `hotpot_qa` loading
script used previously. Fix: `environment.yml` pins `datasets==2.21.0`
explicitly rather than `datasets>=X`.
**Encoded in:** `environment.yml`.

### #2 — Single-seed CV gives noisy, unstable conclusions
**Inherited from the earlier project iteration.** A single 5-fold CV split
produces only 5 AUC values; at small k / small sample sizes, "which k first
crosses a threshold" was effectively determined by which fold seed was
used, not by the underlying signal. Fix: always pool fold-level AUCs across
many seeds (`core.stats.repeated_cv_auc` / `matched_repeated_cv`), and judge
threshold-crossing by the CI LOWER bound, not the point estimate
(`core.stats.ci_lower_bound_exceeds`).
**Encoded in:** `core/stats/multiseed.py`, `core/stats/ci.py`.

### #3 — Feature selection performed on the full dataset before CV split (leakage)
**Inherited from the earlier project iteration.** An earlier version
selected the "top-k" neurons using ALL samples (including ones that ended
up in the test fold), which produced impossibly high AUC (k=512 -> AUC
0.99+) that vanished once selection was correctly confined to each fold's
training portion only. Fix: `core.stats.nested_cv.select_features()` only
ever receives the training-fold data; a deliberately-leaky version
(`cross_validated_auc_LEAKY_FOR_TESTING_ONLY`) is kept ONLY as a regression
test fixture in `experiments/exp_00_sanity_check`, never called from real
experiment code.
**Encoded in:** `core/stats/nested_cv.py`; regression-tested in
`experiments/exp_00_sanity_check/run_stats_sanity_check.py` check (d).

### #4 — HuggingFace endpoint / network configuration issues on AutoDL
**Inherited from the earlier project iteration.** Loading the model via the
HF Hub API on the AutoDL instance ran into endpoint/network configuration
problems. Fix: always load from the local path
(`/autodl-fs/data/Llama-3.1-8B-Instruct`) with `local_files_only=True`,
never fall back to a Hub download.
**Encoded in:** `core/extraction/model_loader.py`.

### #5 — Naive substring matching under/over-counts correct answers
**Inherited from the earlier project iteration.** Strict `reference in
prediction` matching missed correct answers phrased differently and
falsely matched unrelated short references. Fix: word-set overlap matching
in `core.labeling.correctness.is_correct`, with explicit non-goals
documented (no numeric normalization, no synonym resolution) so the
matching logic doesn't silently grow scope creep.
**Encoded in:** `core/labeling/correctness.py`.

---

### #6 — （沙箱环境限制，非真实bug）core/data/loaders.py 尚未在真实数据上跑通
2026-08-11。编写`core/data`时沙箱环境无法访问huggingface.co,所有5个数据
集加载器的字段名是对照HF数据集页面的文档/预览手动核实的,但没有实际执行过
`datasets.load_dataset(...)`。已建立`experiments/exp_01_data_check`作为服
务器端的验证步骤,必须先跑通并确认字段内容合理,再在此基础上写完整的抽取
流水线。若跑的时候CoQA的`trust_remote_code=True`触发了非预期行为或报错,
在这里补充一条新记录说明具体报错和解决方式。

### #7 — GSM8K数值提取正则会误匹配"孤立逗号",导致float('')崩溃
2026-08-12。用户pilot运行时gsm8k数据集shard 4/5报错
`ValueError: could not convert string to float: ''`。根因:
`extract_first_number`/`extract_gsm8k_reference_number`用的正则
`-?[\d,]+(?:\.\d+)?`只要求"一个或多个{数字,逗号}字符",没要求其中必须包
含至少一个数字——模型生成文本里的普通句中逗号(如"Let me think, the
answer is...")会被单独匹配成一次"数字",转换时`","`去掉逗号后变成空字符
串,`float('')`直接抛异常,而不是文档承诺的返回`None`。
修复:正则改为`-?\d[\d,]*(?:\.\d+)?`,强制匹配必须以数字开头(可选负号
后紧跟一个数字),外层再加`try/except`兜底防止类似情况再次以异常形式冒出
来。已在`experiments/exp_00_sanity_check/run_stats_sanity_check.py`里加了
专门的回归测试(check (f)),覆盖了这次报错的原始场景以及其他边界情况
(纯逗号、多个逗号无数字、负数、千分位、小数、空字符串),防止同类正则边
界问题以后再犯。
**编码于:** `core/labeling/gsm8k_correctness.py`;回归测试见
`experiments/exp_00_sanity_check/run_stats_sanity_check.py`。

### #8 — 长时间运行后CoQA个别batch OOM(显存碎片化+长story叠加)
2026-08-12。1000条/数据集的正式粗扫跑完,4817个batch里只有CoQA的20个连续
batch(shard 800-819)报CUDA OOM,报错时PyTorch已占用18.23GB/23.56GB,分配
2.55GB时失败。其余4780个batch(含同样batch_size=1的CoQA前800个)全部成
功,不是batch_size系统性设置过大的问题。可能原因:①这20个shard连续,大
概率是CoQA里同一个story连续几轮问答,story文本本身偏长,叠加10次采样的
KV cache;②长跑几千次generate()调用后显存碎片化累积,刚好在这一段被顶
爆。
修复:①`run.py`每个batch处理完(不论成功失败)都主动调用
`torch.cuda.empty_cache()`,减少长跑累积的碎片化;②失败时把该batch里最
长question的字符数记进日志,便于后续判断是否和长文本相关。
建议下次重跑失败batch时,启动命令前加上环境变量
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,进一步缓解PyTorch自
身提示的碎片化问题。
**编码于:** `experiments/exp_02_coarse_scan/run.py`。若加了这些缓解措施
后同样的shard还是反复OOM,说明是这几条story本身长度就逼近显存上限,需要
单独讨论是否要对CoQA的story长度做额外截断,而不是继续靠碎片整理硬顶。

### #9 — L1特征排序在k值扫描中被重复计算,ffn_neuron(高维)上极慢
2026-08-12。pilot运行卡在`ffn_neuron`(14336维)那个cell上8分钟没有任何输
出,用户反馈"看起来像卡死"。排查后发现不是死锁,是`core/stats`原来的设计
问题:`repeated_cv_auc`对每个k值(32/256)都独立调用一次
`cross_validated_auc`,而`select_features`(method="l1")每次都会在**完整
维度**的训练集上重新拟合一次L1惩罚逻辑回归去排序特征——这个排序结果其实
和k值无关(只是排完序后取前多少个的区别),但同一份数据在同一折里被重复
算了两遍昂贵的全维度拟合。在4096维的`attention`上这个浪费不明显,但在
14336维的`ffn_neuron`上,每次L1拟合本身就慢,乘以"每个k值重跑一次"×
"5 seed×5折=25折"×"2个需要选择的k值",肉眼可见地卡住。
修复:新增`select_features_ranking()`返回完整排序(不截断),
`cross_validated_auc_multi_k()`和`repeated_cv_auc_multi_k()`对一组k值只
在每折里做一次排序、复用给所有k值。`exp_02b_analysis/run.py`改成调用
multi_k版本。同时给`repeated_cv_auc_multi_k`加了`on_seed_done`回调,
`exp_02b_analysis`用它在每个seed跑完后打一行心跳日志,以后即使某个cell确
实要跑几分钟,日志也不会长时间沉默。
**编码于:** `core/stats/nested_cv.py`、`core/stats/multiseed.py`、
`experiments/exp_02b_analysis/run.py`。已在沙箱用假数据验证过multi_k版本
结果和续跑逻辑均正确。

### #10 — run_02_compare.py无续跑机制,2GB CPU实例上高维模块OOM导致进度全部丢失
2026-08-14。`exp_03_posthoc_pooling_test/run_02_compare.py`最初版本没有像
`exp_02b_analysis`那样做增量落盘+续跑,纯CPU分析任务在2GB内存的AutoDL无卡
实例上跑到`ffn_neuron`(14336维)那个cell时被系统OOM killed,前面25分钟、
18个cell的计算全部丢失,需要从头重跑。
根因:①内存太紧,14336维数据的sklearn L-BFGS拟合加上同时持有pregen/
posthoc_last/posthoc_answermean三份激活数组,峰值内存超过2GB;②脚本本身
没有续跑机制,是纯粹的工程疏漏——`exp_02b_analysis`当时已经加了增量JSONL
落盘+resume-skip,这个脚本理应遵循同样标准,当时漏掉了。
修复:①`run_02_compare.py`改成和`exp_02b_analysis`一致的模式,每个cell算
完立刻写入JSONL并flush,重启自动跳过已完成的cell;②每个cell处理完(不论
成功失败)显式清空承载激活数组的dict并调用`gc.collect()`,减少长跑累积的
内存峰值(GPU阶段的`torch.cuda.empty_cache()`在CPU分析阶段的对应做法);
③建议实际运行时把CPU实例内存从2GB调大一档(成本可忽略),从根本上减少
OOM发生概率,不完全依赖代码侧的内存优化。
**编码于:** `experiments/exp_03_posthoc_pooling_test/run_02_compare.py`。
沙箱内用假数据验证过增量写入、续跑跳过在重构后依然正确。

<!-- New entries go below this line. Use the same format: what happened,
how found, the fix, and which file encodes it. -->
