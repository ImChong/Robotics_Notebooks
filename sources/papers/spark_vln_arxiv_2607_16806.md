# SPARK-VLN: Dynamic Social Vision-Language Navigation（arXiv:2607.16806）

> 来源归档（ingest）

- **标题：** Token-Wise Latent Streaming from Slow Reasoners to Fast Planners for Dynamic Vision Language Navigation（项目页简称 SPARK-VLN）
- **类型：** paper / VLN / social-navigation / dual-system / flow-matching / low-latency
- **来源：** arXiv abs / PDF / HTML（v1，2026-07-22）
- **原始链接：**
  - <https://arxiv.org/abs/2607.16806>
  - PDF：<https://arxiv.org/pdf/2607.16806>
  - HTML：<https://arxiv.org/html/2607.16806v1>
  - 项目页：<https://hutslib.github.io/SPARK-VLN.dc.html>
- **作者：** Tianshuai Hu、Yangyi Zhong、Zeying Gong、Lingdong Kong、Xiaodong Mei、Guoyang Zhao、Xiaolu Liu、Song Wang、Rong Li、Junwei Liang（\* 通讯作者）
- **机构：** 香港科技大学（HKUST）；香港科技大学（广州）（HKUST-GZ）；新加坡国立大学（NUS）；浙江大学（ZJU）
- **入库日期：** 2026-09-07
- **一句话说明：** 动态社会 VLN 快慢双系统：慢 VILA-8B 在自回归生成过程中逐 token 流出隐状态，经 Sequence-to-Slot Bridge 压缩为 8 个 latent slot 实时条件化 rectified flow-matching 快规划器；配套「推理时不暂停仿真」的人中心基准，Realistic 环境 VLN SR 34.8%、相对 Wait-then-Act +10 pp；截至入库日项目页 Code 占位、确认未开源。

## 开源状态（核查 2026-09-07）

- **项目页 Code 按钮：** `href="#"` 占位，无 GitHub / Hugging Face / Zenodo 外链。
- **arXiv / PDF：** 未列官方代码或数据下载。
- **结论：** **确认未开源**（无仓库、权重、基准包）；仅项目页与论文可读。
- **互指：** [`sources/sites/spark-vln.md`](../sites/spark-vln.md)；升格实体 [`wiki/entities/paper-spark-vln.md`](../../wiki/entities/paper-spark-vln.md)。未建 `sources/repos/`。

## 摘要级要点

- **问题：** 动态人中心环境中，VLM 语言推理慢、社会合规规划要快；推理期间环境持续演化 → **observation staleness**（观测陈旧），传统 VLN 基准在推理时 **冻结仿真** 掩盖该风险。
- **洞察：** VLM 自回归生成 **尚未结束** 时，中间 token 隐状态已编码方向意图与行人上下文，可被快规划器即时利用。
- **架构（三模块）：** (1) **Token-Wise Hidden Streamer** — 指定 decoder 层（16–23）逐 token 抽取隐状态；(2) **Sequence-to-Slot Latent Bridge** — Perceiver 式 cross-attn 将变长隐流压成 **N=8** 固定 slot；(3) **Evolving Latent Conditioner** — 将 slot 注入 **rectified flow-matching** 专家规划器（5 步 Euler + ESDF 安全 critic 选轨迹）。
- **慢系统：** VILA-8B；输入 8 帧 egocentric RGB + 语言指令。
- **快系统：** PointNav 预训练；VLN 评测时 **无坐标目标**，仅靠 streamed slot 承载语言意图。
- **基准：** 人中心动态社会 VLN suite — **Idealized**（推理时暂停仿真）vs **Realistic**（推理时行人继续移动）；ORCA 行人、密度按场景面积缩放；覆盖 PointNav + VLN；指标含 SR/NE/SPL + PSC/Col/AvgD + **staleness 统计**。
- **主结果（VLN，Table I）：** Idealized SR **41.6%** / SPL 34.67%（超 NaVILA +4.4 pp、DualVLN +15.4 pp）；Realistic SR **34.8%** / SPL 28.68%（超 NaVILA 29.0%、DualVLN 22.9%），碰撞率最低 29.3%。
- **消融（Realistic）：** Stream vs Wait-then-Act SR **+10.0 pp**（34.8 vs 24.8）；Cross-Attn vs Mean-Pool **+9.6 pp**。
- **延迟（Table V，RTX 4090）：** Stream 有效 per-update 延迟 **0.185 s** vs W-t-A **0.788 s**；推理期间行人平均位移 0.050 m vs 0.213 m。

## 核心论文摘录（MVP）

### 1) 问题：observation staleness 与同步 VLN 评测盲区

- **链接：** <https://arxiv.org/abs/2607.16806> §I、§II-C、§V-A
- **摘录要点：** 主流 VLN 在推理时冻结环境，行人不动，latency 不影响结果。动态社会中，慢 VLM 推理窗口内场景已变，「推理开始时安全」的机动到执行时可能碰撞。现有双系统（DualVLN 等）只在 **推理结束** 才向快规划器交接一次 latent，快系统在整段推理中 **无新鲜引导**。
- **对 wiki 的映射：**
  - [SPARK-VLN](../../wiki/entities/paper-spark-vln.md)
  - [视觉–语言导航（VLN）](../../wiki/tasks/vision-language-navigation.md)
  - [具身大模型实时性 ↔ 泛化取舍](../../wiki/concepts/embodied-fm-latency-generalization-tradeoff.md)

### 2) 方法：逐 token 隐状态流 → slot → flow-matching 规划器

- **链接：** §III-C、§IV、Fig. 2–3
- **摘录要点：** 自回归每步 k 抽取多层隐状态 H^(k_t)，Bridge 用可学习 query cross-attn 得固定 R_t，规划器以观测 token + R_t 做 rectified flow，生成 N_c 条候选轨迹，critic 按 ESDF 选最安全。语言目标模式下 z_g = R_t，随 token 生成 **持续刷新**。
- **对 wiki 的映射：**
  - [SPARK-VLN](../../wiki/entities/paper-spark-vln.md)
  - [NaVILA](../../wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md) — 单系统 VILA + 动作解码对照
  - [ABot-N1](../../wiki/entities/paper-abot-n1.md) — 另一慢–快 VLN 双系统路线（像素接口 vs 隐状态流）

### 3) 基准：Idealized vs Realistic 动态社会 VLN

- **链接：** §V-A、Fig. 4、Table I–II
- **摘录要点：** Idealized：有行人但推理时暂停；Realistic：推理时仿真不阻塞。中位起终点距 8.2 m、行人巡逻路径 17.4 m。社会场景含正面相遇、路口、跟随、转角。所有方法从 Idealized 迁到 Realistic 均退化，证明 staleness 是真威胁。
- **对 wiki 的映射：**
  - [SPARK-VLN](../../wiki/entities/paper-spark-vln.md)
  - [HumanoidVLN](../../wiki/entities/paper-humanoidvln.md) — 另一强调物理/社会评测的 VLN 基准对照

### 4) 消融与运行时：streaming 的收益可量化

- **链接：** §V-C、Table III–V
- **摘录要点：** W-t-A 与 Stream 共享同一 VLM 与规划器，仅交接机制不同 → +10 pp SR。Mean-Pool 丢 token 结构 → Cross-Attn +9.6 pp。Stream 延迟 0.185 s 为 VLM 类方法中最低之一，行人位移统计与安全性提升一致。
- **对 wiki 的映射：**
  - [SPARK-VLN](../../wiki/entities/paper-spark-vln.md)
  - [FSD-VLN](../../wiki/entities/paper-fsd-vln.md) — 另一快慢双系统 VLN（空中 / 语义缓冲）

## 对 wiki 的映射（汇总）

- [`wiki/entities/paper-spark-vln.md`](../../wiki/entities/paper-spark-vln.md) — 主实体页
- [`wiki/tasks/vision-language-navigation.md`](../../wiki/tasks/vision-language-navigation.md) — 动态社会 / 延迟感知 VLN 子域
- [`wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md`](../../wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md) — VILA 单系统强基线
- [`wiki/entities/paper-abot-n1.md`](../../wiki/entities/paper-abot-n1.md) — 慢–快像素接口双系统对照
- [`wiki/entities/paper-fsd-vln.md`](../../wiki/entities/paper-fsd-vln.md) — 空中快慢双系统对照
- [`wiki/concepts/embodied-fm-latency-generalization-tradeoff.md`](../../wiki/concepts/embodied-fm-latency-generalization-tradeoff.md) — 异步双频 / token streaming 实例
