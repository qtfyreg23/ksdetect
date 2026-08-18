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

### D11 — posthoc抽取这一轮粗扫用"last token"而不是"仅答案片段mean pooling"
2026-08-11。posthoc本应该更能体现"生成的答案本身"的信号,理想做法是只在
新生成的答案token范围内做mean pooling,但需要精确对齐每条样本答案的token
边界(尤其在重新拼接文本、重新分词之后,边界不一定和`generate()`时的token
切分完全一致)。这一轮粗扫为了控制复杂度和风险,posthoc和pregen一样都用
"最后一个token"的激活,pooling方式在`config.yaml`里是`pooling_posthoc:
"last"`。如果阶段2机制验证阶段发现posthoc信号异常弱,第一件要复查的事就
是这个简化,而不是急着下"posthoc没有信号"的结论。

### D12 — 生成用chat template包装,而不是裸文本续写
2026-08-11。LLaMA-3.1-8B-Instruct是instruct模型,生成阶段统一用
`tokenizer.apply_chat_template`包装成对话格式(单轮user消息)再送入
`generate()`,而不是把question当纯文本续写——否则贪婪生成大概率生成不连
贯的续写而非规范回答。pregen激活抽取用的也是这个chat模板包装后的文本(代
表"模型即将开始回答之前"的状态),posthoc则是"chat模板文本+生成的答案"拼
接后重新分词。

### D13 — 抽取流水线的续跑机制:标签shard存在即视为该batch已完成
2026-08-11。`exp_02_coarse_scan/run.py`每个batch处理完,先落盘activation
shard,最后落盘标签shard(JSONL)——标签shard文件是否存在,就是续跑时判断
"这个batch要不要跳过"的唯一依据,顺序不能颠倒(必须最后写标签文件,否则
中途失败会把没抽完activation的batch误判为已完成)。同时给每次运行的
`output_dir`绑定一个config内容的hash值,复用同一个`output_dir`但config变
了会直接报错拒绝启动,防止新旧设置的结果混在一起(对应project plan
§2.3)。

### D14 — 粗扫规模下调:batch_size强制降到1,n_examples从2000降到1000
2026-08-12。pilot(8条/数据集)实测:batch_size=4在24GB显存下OOM(有效生
成batch是`batch_size × n_samples_for_soft_label`=4×10=40条序列同时生
成,超出显存),用户手动降到batch_size=1才跑通。按batch_size=1的实测速率
换算,原计划2000条/数据集(TruthfulQA除外用817全量)预计总耗时约17.3小
时,其中MedQuad和GSM8K(生成长度256 token)占了约75%的耗时。与用户确认
后,统一把`n_examples_per_dataset`降到1000,预计总耗时降到约8.5小时。
`config.yaml`和`config_pilot.yaml`的`batch_size`默认值都改成1(不再是
4),避免任何人从头跑一遍时重新踩一次OOM。若后续想恢复更大规模,需要重新
挂一轮pilot观察显存,不能直接改回2000/更大batch_size。

### D15 — soft_binarized标签:用soft_label > 0作为二值化阈值
2026-08-12。`exp_02b_analysis`里`repeated_cv_auc`需要二值标签,但软标签
(soft_label)是连续值(10次采样里的错误率)。这一轮粗扫分析把软标签简单
二值化成`soft_label > 0`(即"10次采样里只要错1次就算不可靠"),作为硬标
签之外的第二种`label_type`并行跑,不代表这是"正确"或唯一合理的阈值——只
是粗扫阶段为了让软标签也能纳入AUC网格比较而选的一个简单、可复现的默认
值。如果后续现象筛选阶段发现软标签本身值得深挖,阈值选择需要专门讨论(比
如改成>=0.5多数错误,或者不二值化改用其他连续目标的分析方式)。

### D16 — embedding模块在分析网格里固定用layer=-1,不套用其余9层
2026-08-12。`core/extraction/hooks.py`里embedding只在每次前向传播时被记
录一次(不分层深度),固定存在`layer_-1`下。`exp_02b_analysis/run.py`最初
的cell构造逻辑对所有模块统一套用`layers_coarse`这9个层,会导致embedding
在8/9个层上找不到对应shard(全部报FileNotFoundError,虽然被正确捕获跳
过,但产生大量无意义的错误日志、浪费查找时间)。修复:新增
`layers_for_module()`函数,embedding固定只返回`[-1]`,其余模块正常套用
`layers_coarse`列表。这个bug是在用假数据做端到端沙箱测试时发现的,发现即
改,未影响任何已跑的服务器端任务。

### D17 — 新增answer_mean pooling,用于阶段3的posthoc<pregen现象区分实验
2026-08-13。阶段1粗扫发现TruthfulQA/MedQuad的posthoc AUC反而低于pregen,
在D11里已经标注过posthoc用"last token"pooling是粗扫阶段的简化,现在这个
简化成了需要检验的对象。给`core/extraction/extract.py`的`extract_batch`
新增`pooling="answer_mean"`选项(只在生成答案的token范围内做mean
pooling,不含prompt部分),配合新增的`compute_prompt_token_lengths`(逐条
非padding分词,得出每条prompt的真实token数,从而推算出posthoc文本里答案
部分的token数)。索引逻辑(左padding场景下"取最后N个真实token"是否有偏移
错误)已经用纯numpy单独验证过,沙箱里也用假数据端到端跑通了
`exp_03_posthoc_pooling_test`的比较脚本(含对齐逻辑、embedding自动排除、
判定规则)。这个pooling方式只用于`exp_03`这个针对性实验,**没有**回填进
`exp_02_coarse_scan`的默认配置——是否要把粗扫的posthoc pooling方式整体换
成answer_mean,要等这次判定结果出来再讨论,不能因为"感觉应该更好"就直接
改掉已经产出的阶段1数据。

### D18 — exp_03两阶段判定结果(2数据集试跑)与推广到全部5数据集的决定
2026-08-14。`exp_03_posthoc_pooling_test`对TruthfulQA(全部完成)+MedQuad
(因OOM中断)的判定结果:18个已完成cell里16个`A_supported`、2个`mixed`、
0个`B_supported`,`posthoc_answermean`相对`posthoc_last`的提升几乎全部
达到p<0.0001的显著水平。**结论:阶段1粗扫里"posthoc比pregen弱"这个现
象,主要是"last token pooling+固定chat模板后缀"导致的测量artifact,不是
真实的表征层面masking效应——原来的现象假设(解释B)不成立,予以撤回。**
由于这个pooling问题是聊天模板机制本身导致的、和具体数据集内容无关,判断
它很可能系统性影响了阶段1粗扫全部5个数据集的posthoc数据,不只是
TruthfulQA/MedQuad这两个被挑出来细查的。决定:把`exp_03`的
answer_mean重抽+三方比较推广到全部5个数据集(`config_full_rescan.yaml`,
使用全新的`output_dir`/`activation_dir`,不与2数据集试跑的结果混用),
目的是订正整张阶段1宽扫描地图的posthoc部分,而不是仅针对这两个数据集打
补丁。订正后需要重新审视全貌,原本"哪些现象值得深挖"的判断可能因为这次
订正而改变。

### D19 — config-hash防护逻辑从exp_02内联代码提取为core.run_utils共享工具
2026-08-14。`exp_03`的`run_01`/`run_02`都需要和`exp_02`同样的"复用
output_dir时配置变了就报错拒绝启动"这个防护逻辑——这是第三次要写同一段代
码,按项目规则(§2.5)不应该再复制粘贴,提取成`core/run_utils.py`。已验证
新函数的哈希结果和`exp_02`原来内联实现逐字节一致,`exp_02`已经产出的
`config_hash.txt`不会因为这次重构失效。

### D20 — 全量5数据集判定结果:MedQuad是唯一真实效应,其余4个是artifact
2026-08-14。`exp_03`全量5数据集重扫结果:coqa(33A/2mixed/1B)、gsm8k
(36A)、triviaqa(36A)、truthfulqa(34A/2mixed)——四个数据集几乎全部判A,
且订正后posthoc_answermean均值普遍反超pregen,原来阶段1粗扫看到的
"posthoc偏弱"确认是pooling artifact,这四个数据集的该现象撤回。
**MedQuad是唯一例外**:36个cell里只有8个判A,17个mixed、11个B,gap稳定在
0.01-0.06,p值普遍<1e-4,残差/FFN模块尤其明显。这是一个真实、跨模块跨层
一致的效应,不是噪声。**决定**:把原来"TruthfulQA+MedQuad"两数据集合并
的现象假设拆开——TruthfulQA部分作废,MedQuad部分作为新的、更聚焦的候选现
象保留,进入下一步排查。

### D21 — 引入解释C(浅层捷径混淆),在深挖机制前先排除
2026-08-14。MedQuad的hard_label分布极端不平衡(950/1000即95%判错,来自
`run_00_qualitative_check`早前的定性预检),这本身可能是另一套解释:
pregen阶段的高AUC,可能不是读出了"模型是否真的知道答案"这种深层知识信
号,而是读出了问题文本本身某种浅层捷径(比如题目对应的疾病是否常见/题目
表述长度等),而posthoc生成的大段套路化疾病描述冲淡了这种捷径。新增
`run_03_medquad_confound_check.py`,用问题的词数/字符数这类最廉价的表层
特征单独跑`core.stats.repeated_cv_auc`,如果这类特征本身就能获得接近
pregen真实水平的AUC,说明捷径解释成立,需要在动"fluency masking"这类机
制故事之前先把这个排除掉。已在沙箱用两种极端假数据(长度与标签无关/强相
关)验证判定逻辑两个方向都正确响应。

### D22 — H1/H2区分实验:新增前K窗口pooling,以及跨数据集普适性预检
2026-08-15。在MedQuad现象(D20/D21)基础上,讨论了H1(稀释:信息集中在答
案前部,整段平均稀释信号)和H2(内容/文体掩盖:模型一开口这件事本身盖住
信号,与位置无关)两种机制假说的普适性——H1本质是"critical token定位"
这条已有研究路线的一个实例,天花板有限;H2如果成立,能预测"post-hoc检测
方法在哪些领域系统性更弱",可以直接对照LAFaCT等已发表工作的跨domain数
字做外部验证,也能直接指导"根据领域文体同质性自适应选择依赖pregen还是
posthoc信号"这样的方法设计,普适性和方法论价值明显更高。
决定分两步验证:①`exp_04_cross_dataset_check`(纯CPU,零成本):用"正确
/错误答案长度分布的Cohen's d"作为文体同质性的粗糙代理,和已知的
pregen-posthoc gap跨5个数据集做相关性检查(n=5,仅作方向参考,非确证)。
②`exp_05_medquad_dilution_vs_style`:给`extract_batch`的`answer_mean`
pooling新增`answer_window_k`参数(只取答案开头前K个token,不是整段),
对MedQuad(主要)和TruthfulQA(对照,已知是纯artifact数据集)重新抽取
K=20/50/100三档,和已有的pregen/posthoc_last/posthoc_full做六方配对比
较,判定规则(front20是否显著弱于full)已用两种方向相反的构造数据验证过
两个分支都正确。

### D23 — H1/H2判定的自动verdict函数太粗,漏掉了真正的模式;MedQuad现象重新定性
2026-08-16。`exp_05`全量结果出来后,自动`verdict()`函数把绝大多数cell判
成`H2_supported`(69/72),但人工核对均值发现函数没抓住真正的规律:
`front50`(0.836)、`front100`(0.837)几乎和`pregen`(0.839)完全一致(四个
模块的gap都在-0.014~+0.019之间,基本是噪声),只有`full`(整段平均,
0.795)明显更差——这是"先升后降"的非单调模式,不是我原来预设的"单调回
升(H1)"或"全程持平(H2)"任何一种,自动判定函数只比较了front20 vs
full,没有捕捉到front50/100这个更关键的中间态。

**结论重新定性**:MedQuad答案特别长(平均125-157词),选对pooling窗口
(前50-100个token)之后,posthoc基本追平pregen——**MedQuad的"posthoc更
弱"现象,大概率也是pooling窗口和答案长度不匹配导致的artifact,不是真实
的表征层面masking效应(H2)。** 结合`exp_04`跨数据集检验里长度同质性代理
和gap的相关性也只有0.10(接近0,不支持H2的普适性预测),H2这条"fluency
masking"的机制故事现在证据不足,不建议继续按这个方向包装成论文的核心贡
献。

**教训记录**:自动化的二元判定函数(`H1_supported`/`H2_supported`)在设
计时只考虑了两种预设模式,遇到真实数据里更复杂的非单调模式时会给出误导
性结论。以后类似的多档位扫描,除了看自动判定,必须把每一档的原始均值列
出来人工过一遍,不能只信一个自动化标签。已补充
`run_03_optimal_window_check.py`直接检验front50/100是否和pregen统计上不
可区分。

### D24 — run_03结果修正:逐层看front50/100不是干净地追平pregen,有显著双向波动
2026-08-16。`run_03_optimal_window_check.py`全量结果:模块平均层面确实
front50/100比full更接近pregen(方向判断不变),但逐层看,front50显著偏离
pregen的cell有47/72(65%),而且有的层front50显著低于pregen(如MedQuad
attention层0:0.813 vs 0.850,p=0.0002),有的层front50显著高于pregen
(如MedQuad attention层12:0.867 vs 0.781,p<1e-13)。**教训**:之前只看
模块平均值下的"gap≈0"结论,把这种层间双向波动平均掉了,显得比实际更干
净。这不改变"MedQuad现象主要是pooling窗口artifact、H2证据不足"这个大方
向判断,但后续涉及"跨层平均相关系数"的分析,必须同时附上逐层原始数据,
不能只报一个汇总数字。

### D25 — exp_06:用订正pooling复核层深度依赖模式,只需给GSM8K补抽
2026-08-16。決定重新检验阶段1粗扫里"层深度依赖任务类型"这个候选现象(深
度敏感:TriviaQA/CoQA/TruthfulQA(posthoc);深度不敏感:MedQuad/GSM8K),
因为原始数据用的是有问题的last-token pooling,不能排除这个模式本身部分
是artifact。按平均答案长度判断,只有MedQuad(已在exp_05订正)和GSM8K
(120+词,需要新抽)会因为pooling窗口选择产生实质差异,TruthfulQA
(exp_05已订正)、TriviaQA/CoQA(答案很短,front100约等于exp_03已有的整
段平均)不需要新抽取。同时按D24的教训,`run_02`的输出强制附带逐层原始数
据,不能只依赖汇总相关系数下结论。`run_03`把exp_03的MedQuad长度捷径检验
泛化到GSM8K,并针对数学题额外加了"问题里数字个数/最大数字"这两个候选捷
径特征,已用构造数据验证检出逻辑正确。

### D26 — exp_06订正结果:TruthfulQA的posthoc"深度依赖"撤回,MedQuad浅层饱和排除长度捷径,GSM8K呈非单调模式
2026-08-19。`exp_06_depth_correlation_recheck`全量结果(corrected_depth_results.csv,
depth_correlation_comparison.csv,shallow_shortcut_audit.json)核对完毕:
1. CoQA(0.885/0.434)、TriviaQA(0.805/0.664)的深度依赖模式完整保留,订正
   前后基本不变。
2. **TruthfulQA的posthoc"深度依赖"成员资格被撤回**:相关系数从订正前0.793
   跌到订正后-0.038,和D18里TruthfulQA/MedQuad最早被识别为纯artifact性质
   一致——这次是posthoc"深度依赖"分组本身的假象,不是新发现,是旧问题的
   延伸清理。
3. **MedQuad浅层饱和确认为真实效应,非长度捷径**:pregen/posthoc相关系数
   均在0.02-0.18,run_03捷径审计显示word/char length的AUC只有0.54-0.59,
   远低于真实的0.83-0.87,长度解释(D21的泛化检验)被排除。
4. **GSM8K是先升后降的非单调模式,不是单纯"深度不敏感"**:layer0~0.60,
   layer12峰值~0.68,layer31回落到~0.63,线性相关系数(0.21/0.40)因此被
   低估,不能直接读作"和MedQuad一样浅层饱和"。同时捷径审计显示GSM8K
   layer0的AUC(0.591-0.606)和"文本长度+数字个数"捷径合并特征的AUC
   (0.599,CI[0.590,0.608])统计上无法区分,但layer12峰值明显超出这个
   捷径天花板——浅层可能主要是捷径,深层有真信号在往上加。
5. （附带发现,零成本从上述CSV直接算出,未做新实验）layer0起点AUC和后续
   深度增益之间存在中等强度负相关(raw空间-0.667,logit空间-0.570,
   n=10个数据集×stage点,MedQuad和TriviaQA-pregen是两个主导相关性的极
   端点)——"有没有空间涨"这个更简约的统一解释值得记录,但样本点太少、
   被两个极端值主导,暂不单独立项验证,留待后续有更多数据集/stage点时
   再检验是否稳健。

**决定**:阶段3聚焦MedQuad的浅层饱和现象(第3点)作为最干净、最站得住的
候选现象,新增`exp_07_medquad_shallow_signal_probe`去区分"这个浅层信号
到底是粗粒度的问题类型/模板分桶,还是更细粒度、无法用简单表层特征复现
的东西"。GSM8K的先升后降模式(第4点)记录在案,暂不单独立项——它不像
MedQuad那样干净,且GSM8K本身已经确认有真实的深层信号增量,不是这一轮
"哪个现象最值得深挖"的优先候选。

<!-- New entries go below this line. -->
