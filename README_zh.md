# ksdetect

知识充分性检测（Knowledge Sufficiency Detection）——基于内部表征的幻觉 /
知识充分性检测研究代码库，按照项目重建实施计划（`研究项目重建实施计划.md`）
重新搭建。请先阅读该文档以了解研究流程（第 0-6 阶段）；本 README 仅涉及
工程层面的内容。

如果下面的任何内容存在歧义，或你不确定某个步骤是否适用于你当前的情况，
请**先停下来询问，再继续**（项目计划 §2.1）——不要靠猜测继续往下做。

---

## 1. 环境搭建

**服务器：** AutoDL，项目目录 `/root/autodl-tmp/ksdetect`。
**模型：** LLaMA-3.1-8B-Instruct，固定本地路径
`/autodl-fs/data/Llama-3.1-8B-Instruct`（如需更改，必须先在
`docs/decisions.md` 中新增一条记录，不能直接改）。
**连接方式：** PyCharm SSH 远程解释器，指向下面创建的 conda 环境（**不是**
本地 venv，也**不是**系统 / base conda 环境）。

### 1.1 创建 conda 环境（在服务器上，通过 SSH 终端执行）

```bash
cd /root/autodl-tmp/ksdetect
conda env create -f environment.yml
conda activate ksdetect
```

如果是重新运行（环境已存在，需要更新）：

```bash
conda env update -f environment.yml --prune
```

**如果 `torch==2.4.1`（CUDA 12.1 版本）安装失败，或运行时报 CUDA 版本不
匹配的错误：** 在服务器上运行 `nvidia-smi`，查看右上角显示的 CUDA 版本，
然后到 https://pytorch.org/get-started/previous-versions/ 找到匹配的
torch wheel 并安装该指定版本——**不要**把 `environment.yml` 里的版本号
放宽成区间（`>=`），而是替换为正确的具体版本，并在 `docs/decisions.md`
中记录这次改动。

### 1.2 让 PyCharm 指向这个环境

1. PyCharm → Settings → Project → Python Interpreter → Add Interpreter →
   On SSH。
2. 输入 AutoDL 的 SSH 连接信息（主机/端口/用户名——从 AutoDL 实例页面
   获取）。
3. 询问解释器路径时，使用该 conda 环境的 python：
   `/root/miniconda3/envs/ksdetect/bin/python`（如果你的 AutoDL 镜像
   使用的 miniconda/anaconda 路径前缀不同，请相应调整——`conda activate
   ksdetect` 之后可以用 `which python` 确认）。
4. 将远程项目路径设置为 `/root/autodl-tmp/ksdetect`，并同步。

### 1.3 验证模型路径

```bash
ls -la /autodl-fs/data/Llama-3.1-8B-Instruct
```

应该能列出模型的配置/权重文件。如果这个路径不存在或是空的，请先在这里
停下来解决——每个实验的配置文件都假定这个路径已经是正确且已填充好的。

---

## 2. 目录结构

```
ksdetect/
├── core/                  # 稳定层 —— 见下方规则
│   ├── extraction/        # 模型加载、forward hooks、批量提取
│   ├── labeling/          # 正确性判断、硬/软标签构建
│   ├── stats/              # 嵌套交叉验证、多随机种子汇总、置信区间、显著性检验
│   └── viz/                 # 可发表质量的绘图（仅限英文）
├── experiments/            # 探索层 —— 每个实验一个文件夹
│   ├── _template/           # 复制这个来创建新实验
│   └── exp_00_sanity_check/ # 基础设施验证，最先运行这个
├── docs/
│   ├── known_issues.md      # 问题记录 —— 排查看起来眼熟的问题之前先查这里
│   └── decisions.md         # 不应被悄悄推翻重议的决定
├── environment.yml
└── README.md                 # 本文件
```

**不可协商的规则：** `experiments/` 下的任何实验脚本**只能**调用
`core/*` 来完成交叉验证、特征选择、显著性检验、标注、模型加载、hook
注册或绘图。如果你发现自己正要写一个实现交叉验证的 `for` 循环，或是
一个包含中文字符串的绘图调用，又或是手写 hook 代码去直接调用
`model.forward()`——请停下来：这些逻辑应该放进 `core/`，而不是写在
实验脚本里。这正是为了避免同一个 bug 将来要在五个不同的地方分别修一遍。

---

## 3. 运行一个实验

### 3.1 第一次运行：健全性检查（sanity check）

```bash
conda activate ksdetect
cd /root/autodl-tmp/ksdetect

# 第一部分 —— 不需要 GPU，先运行这个
python experiments/exp_00_sanity_check/run_stats_sanity_check.py

# 第二部分 —— 需要 GPU + 真实模型
python experiments/exp_00_sanity_check/run_extraction_smoke_test.py
```

关于每一项检查的含义以及需要反馈什么内容，请见
`experiments/exp_00_sanity_check/README.md`。**在两部分都报告 PASS
之前，不要进入任何其他实验。**

### 3.2 运行其他实验

```bash
conda activate ksdetect
cd /root/autodl-tmp/ksdetect
python experiments/exp_<name>/run.py --config experiments/exp_<name>/config.yaml
```

每个实验都会写出：
- `experiments/exp_<name>/run_log.txt` —— 完整执行日志
- `experiments/exp_<name>/results/summary.json` —— 机器可读的结果
- `experiments/exp_<name>/results/*.pdf` —— 图表（仅限英文，按照项目
  计划 §4.5 的要求）

---

## 4. 创建一个新实验

1. `cp -r experiments/_template experiments/exp_YYYYMMDD_short_name`
2. 填写新的 `config.yaml` 中每一个 `REPLACE_ME` 字段——如果有遗漏，
   runner 会拒绝启动。
3. 填写 `run.py` 中带编号的 `TODO` 部分——分别对应：数据集加载、
   模型加载（通过 `core.extraction`，不要重新实现）、激活值提取与分片
   保存、标签构建（通过 `core.labeling`）、统计分析（通过
   `core.stats`）、绘图（通过 `core.viz`），最后写出
   `results/summary.json` + `results/run_report.md`。
4. 运行前：对照最近一次讨论/确定的内容重新检查一遍配置（项目计划
   §2.1 的"确认最新要求已经完整对齐"）——一个技术上能跑通但内容过时的
   配置仍然是错的。
5. 先在**小规模**上运行（`config.yaml` 中的 `n_examples` 设置为较小
   的值），再进行全量运行（项目计划 §2.2）。
6. 小规模运行结果没问题后再扩大规模。在增大 batch size / 并发之前，
   先在小规模运行时观察 GPU 显存/吞吐情况（项目计划 §2.2，关于并发的
   注意事项）。

---

## 5. 向我反馈结果

每次运行之后——无论是成功、部分成功还是失败——请回复以下内容：

1. **实验路径**，例如 `experiments/exp_20260101_layer_scan/`
2. **使用的配置**（文件本身，或与上一次运行相比的差异 diff）
3. **运行状态**：已完成 / 仍在运行 / 出错 / 尚未运行的任务数量。
   如果有任何报错：报错信息，以及是否已经修复并重新运行过（项目计划
   §2.4——不要只报告数量，也要说明正在采取什么措施）
4. **核心结果**：关键数字要带上标准差/置信区间，而不仅仅是点估计值
5. **图表路径**
6. **任何新出现的、意料之外的情况**，以及是否已经记录进
   `docs/known_issues.md`

如果某个结果看起来异常（例如 AUC 异常高/低，或曲线看起来不稳定），
请明确指出，而不要只报告最终数字——异常情况应该首先作为一个流程问题
来排查（"是不是哪里泄漏了/某个 fold 是不是损坏了/样本量是不是太小
了"），而不是一上来就归因于"随机性"或"模型行为"（项目计划 §2.4）。

---

## 6. 出问题时怎么办

1. 不要只是记下"失败了"然后继续——要弄清楚原因。
2. 先查 `docs/known_issues.md`；这个问题可能已经被记录过，并且有
   已知的修复方法。
3. 修复之后：重新运行所有失败的部分（不要在一批任务里留下空缺），
   如果是一种新的失败模式，请按照现有条目的格式在
   `docs/known_issues.md` 中新增一条记录。

---

## 7. 已知问题 / 决定记录

见 `docs/known_issues.md` 和 `docs/decisions.md`。开始新工作之前先
浏览一遍这两个文件——它们记录了其他地方不会重复说明的上下文信息。
