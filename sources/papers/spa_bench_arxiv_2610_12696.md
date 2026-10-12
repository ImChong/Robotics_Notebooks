# Spa-Bench 论文来源归档

## 元数据

- **题目：** Spa-Bench: A Real-Robot Benchmark for Spatial Grounding in Vision-Language-Action Models
- **方法：** Spa-Bench（真机空间落地基准，非新策略）
- **作者：** Justin Tien-Smith, Matthew Pidden（共同第一作者）, Rolandos Alexandros Potamias（指导）
- **单位：** Imperial College London
- **预印本：** https://arxiv.org/abs/2610.12696
- **HTML：** https://arxiv.org/html/2610.12696v1
- **版本：** v1，2026-10，cs.RO
- **项目页：** https://justintiensmith.github.io/Spa-Bench/（归档见 [justintiensmith-github-io-spa-bench.md](../sites/justintiensmith-github-io-spa-bench.md)）
- **数据：** https://huggingface.co/datasets/Spa-Bench/spa-bench-data-information（仅核对到元数据表）
- **代码：** 项目页未列 GitHub 仓库
- **入库日期：** 2026-10-12
- **读取范围：** arXiv HTML 正文与表格（WebFetch 抽取）。

## 论文要解决的问题

VLA 评测通常只报聚合操作成功率，混合了物体识别、语言理解、空间落地与运动执行。论文问：当前 VLA 能否把空间描述可靠地转成真机行为？核心主张：被测 VLA 往往能完成所需操作，但缺乏稳健的空间落地。

## 基准设计

- **平台：** SO-101 机械臂，桌面场景，共用碗形容器，统一抓放动作族；同一房间、固定光照与相机。
- **六项任务：** Object State、Relative Size（属性）；Referential Description、Relational Placement（关系）；Ordinal Position、Counting（场景结构）。27 个可移动物体实例，11 类形态族。
- **留出单元划分（withheld-cell）：** 每个“概念–参数”组合在空间落地训练中被留出，但其组成部分（物体、概念、动作）在训练中其他地方出现，例如“upside-down white cup”被留出，而 “upside-down” 与 “white cup” 均见过。
- **诊断：** 直接点名目标的操作对照、两类指令的改写、跨任务物体试验、无需干预场景（Relative Size 除外）、Relational Placement 的纠正性移动对照。
- **数据量：** 1,200 条人类遥操作示范（每任务 200，约 5.7 小时、612,733 帧；记录五路 RGB，策略用中间与腕部视角）；2,892 个评测回合（三个完整评测策略各 900，两个部分基线共 192）；总计 4,092 个 episode。
- **评测协议：** 每任务每策略 20 ID + 50 OOD + 20 配对直接 + 20 跨任务物体 + 20 无干预；每回合 60 s（VLA-0 为 90 s）；二值成功，Wilson 95% 区间。

## 被测策略与训练

| 策略 | 说明 | 范围 |
|------|------|------|
| MolmoAct2 | 流匹配动作专家，动作 horizon 30 | 完整（900） |
| π0.5 | 连续动作，horizon 50 | 完整（900） |
| GR00T-N1.7 F-LLM | 冻结 LLM，horizon 16，EMA 平滑 | 完整（900） |
| GR00T-N1.7 F-FT | 全量微调 | 部分：120 个 ID 回合（25/120） |
| VLA-0 | Qwen2.5-VL-3B，文本动作 | 部分：72 个 ID 回合（0/72） |

所有策略 12 epoch、全局 batch 96、4 块 NVIDIA GH200，同一批 1,200 示范，取第 12 epoch 检查点；AdamW，学习率因策略而异（如 MolmoAct2 基础 1e-5、π0.5 2.5e-5）。GR00T F-LLM 与 VLA-0 做了运动裁剪（去掉 6.91% 帧，共 570,386 帧）；GR00T 用系数 0.25 的因果 EMA 平滑。F-FT 在自主执行前有未记录的小幅上移，论文估计平均约 10 mm。

## 论文报告的结果

- **ID（共 120）：** MolmoAct2 88/120（73.3%），π0.5 82/120（68.3%），GR00T F-LLM 81/120（67.5%）。
- **OOD（共 300）：** MolmoAct2 156/300（52.0%），F-LLM 133/300（44.3%），π0.5 122/300（40.7%）。
- **配对场景 OOD 空间 vs 直接点名：** MolmoAct2 60.8% vs 89.2%（差 28.3 点）；π0.5 45.0% vs 83.3%（38.3）；F-LLM 41.7% vs 81.7%（40.0）。
- **改写诊断（60 场景）：** MolmoAct2 空间 65.0%→50.0%、直接 90.0% 不变；π0.5 空间 48.3%→41.7%；F-LLM 空间 48.3%→38.3%、直接 81.7%→70.0%。
- **分任务 OOD（每项 50）：** Counting 三策略均 0/50；Object State MolmoAct2 38、π0.5 18、F-LLM 17；Relational Placement F-LLM 39、π0.5 36、MolmoAct2 25；Ordinal Position 27/23/21。
- **无干预场景（四个非关系任务，80 次）：** 聚合 20.0–23.8%；Counting（目标已满足）80–85%，Ordinal Position 与 Object State 全为 0%。Relational Placement 目标已满足时 75–95%，需纠正性移动时 5–15%。
- **失败归因（OOD）：** Counting 148/150 次失败为无动作；Ordinal Position 74/79 选错物体；Object State 75/77 类别对但实例错。

## 复现与解释边界

- 作者自述限制：单一本体/工作空间/物体清单/操作族；每场景至多 5 个候选物体；复位非像素级一致；无干预条件为零样本；直接点名与改写条件的语言输入并不完全相同；跨任务对比为描述性；结果针对完整适配后的策略系统，不能归因到单个架构部件。
- 五策略中仅三项完成全协议，不宜混作统一排名。
- 数据集页面只核对到 CSV 元数据（`model_evaluation` 约 2.89k 行、`training_data` 约 1.2k 行），许可与视频文件未核实。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-spa-bench-spatial-grounding-vla.md
- 汇总页：wiki/overview/embodied-station-34-papers-spatial-grounding-technology-map.md
- 周报来源：[wechat_embodied_station_34_papers_spatial_grounding_2026-10-12.md](../blogs/wechat_embodied_station_34_papers_spatial_grounding_2026-10-12.md)

## 一手来源

- <https://arxiv.org/abs/2610.12696>
- <https://arxiv.org/html/2610.12696v1>
- <https://justintiensmith.github.io/Spa-Bench/>
