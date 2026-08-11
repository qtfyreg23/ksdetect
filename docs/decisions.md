# Decisions Log

Record of decisions that were made deliberately and should not be silently
re-litigated by a future experiment without discussion first. Append-only.

---

### D1 — Project rebuilt from scratch; conclusions not reused, engineering lessons are
Date: project restart. The previous project's specific findings (FFN vs
Residual strength, which layer is the "peak", the H1/H2/H3 hypothesis
chain) are NOT carried into this codebase as assumptions — they must be
re-derived from data if they reappear. Engineering lessons (see
`known_issues.md`) ARE carried forward.

### D2 — Both pre-generation and post-hoc are in scope for the wide scan
The earlier project abandoned pre-generation framing under external
pressure (chasing a SOTA benchmark) rather than because data ruled it out.
`stage` (pregen/posthoc) is a free variable in the Stage-1 wide scan, not
decided in advance.

### D3 — Model: LLaMA-3.1-8B-Instruct, fixed local path
`/autodl-fs/data/Llama-3.1-8B-Instruct`. Not re-litigated per experiment;
change only via an explicit new Decisions entry.

### D4 — Environment: conda, Python 3.10, PyTorch 2.4.1 (CUDA 12.1)
Chosen for broad compatibility with common AutoDL GPU images. If the
server's actual CUDA driver version doesn't match, re-pin `torch` to the
matching wheel from pytorch.org rather than loosening the version pin to a
range — see `environment.yml`'s comment on this.

### D5 — All paper-bound figures are English-only, produced via `core.viz`
Target venue is an international journal/conference. No exceptions, no
per-experiment opt-out. See `core/viz/style.py`'s `assert_ascii_only`.

### D6 — Project directory on the server
`/root/autodl-tmp/ksdetect`. All paths in configs and README are relative
to this root.

---

### D7 — 阶段1粗扫的数据集:与LAFaCT论文附录C的5个QA基准保持一致
2026-08-11。核对LAFaCT原文(ACL 2026 long paper)附录C后确认: TruthfulQA、
TriviaQA(rc.nocontext子集)、CoQA、MedQuad、GSM8K。选择理由: (1)与用户
决定一致; (2)天然覆盖短答案QA/多跳性质较弱但常识性强的QA/对话QA/医学QA/
数学推理五类差异明显的任务; (3)后续阶段6如果要对标LAFaCT的leave-one-out
基准(73.66 AUROC),数据集一致可以直接复用,不需要重新猜测协议。
标注方式(仅供参考,阶段1粗扫不强制复刻): TruthfulQA用GPT标注、TriviaQA
精确匹配、CoQA/MedQuad用AlignScore阈值0.3、GSM8K数值精确匹配——本项目阶段
1统一用`core.labeling`里的word-set匹配(GSM8K例外,用专门的数值匹配),
AlignScore暂不实现,若后续需要跟LAFaCT数字直接对比,需要补上。

### D8 — MedQuad使用`lavita/MedQuAD`这个HF版本
2026-08-11。MedQuad原始论文(Ben Abacha & Demner-Fushman, 2019)数据集在
HF上没有统一权威账号托管,存在多个不同用户上传、字段不完全一致的版本。选
`lavita/MedQuAD`是因为行数(47.4k)与原论文声称的47,457条QA对完全吻合,
字段结构(question/answer/question_type等)清晰完整,已通过数据集页面核
实(非凭记忆假设)。

### D9 — CoQA在阶段1粗扫中被展开为单轮QA,不保留对话历史
2026-08-11。CoQA原始结构是"一个story+多轮问答",为了在阶段1宽扫描里和其
他数据集用统一的(question, references)格式处理,`core/data/loaders.py`
的`load_coqa`把每一轮问答拆成独立样本,上下文只保留story本身,不包含之前
轮次的问答历史。这是阶段1的简化处理,如果后续机制验证阶段发现CoQA的多轮
对话依赖性很重要,需要专门讨论后再改这个loader,不能默认沿用。

### D10 — 数据集本地化:先用`save_to_disk`落盘,加载器只读本地磁盘
2026-08-11。原方案是加载器直接调用`datasets.load_dataset(...)`联网拉取,
用户要求改为先显式下载到本地。改动:新增`scripts/download_datasets.py`,
把5个数据集完整原始数据(所有split)用`save_to_disk`存到
`data/raw_datasets/{name}/`(路径唯一定义在`core/data/paths.py`);
`core/data/loaders.py`全部改为`load_from_disk`读本地,读不到时报错并提示
运行下载脚本,不做静默网络回退。理由:(1)和模型固定本地路径加载
(`MODEL_PATH`,见D3、known_issues #4)的思路保持一致;(2)避免每次跑实验
都产生网络依赖;(3)`trust_remote_code=True`(CoQA需要)只在下载脚本这一
处执行一次,不需要每次读数据都信任远程代码。`data/`目录已在`.gitignore`
里,不进版本控制。

<!-- New entries go below this line. -->
