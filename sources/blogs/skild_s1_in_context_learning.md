# Introducing S1: In-Context Learning for Robotics（Skild AI）

> 来源归档（blog / 公司官方）

- **标题：** Introducing S1: In-Context Learning for Robotics
- **类型：** blog
- **作者 / 组织：** Skild AI
- **原始链接：** <https://www.skild.ai/blogs/s1>（引用块亦写 `https://skild.ai/blogs/s1`）
- **发表日期：** 2026-08（Citation：August 2026）
- **入库日期：** 2026-08-26
- **抓取方式：** 官方页直连（WebFetch）
- **一句话说明：** Skild 旗舰操作基础模型 **S1** 从预训练起就把任务指定为 **上下文视频示范**（非语言），宣称单条视频即可执行 **最长约 10 分钟、预训练未见** 的操作任务，无需后训练；内部对照显示未见任务上 ICL 相对语言条件 VLA 约 **7×** 成功率（100k 小时档：66% vs 9%）。

## 开源 / 项目页核查（步骤 2.5）

| 项 | 结论（截至 2026-08-26） |
|----|-------------------------|
| 项目页 | **无**独立 `*.github.io` 研究项目页；入口为公司博客 + [skild.ai](https://www.skild.ai/) |
| 代码 / 权重 | **确认未开源** — [github.com/skild-ai](https://github.com/skild-ai) **0 个公开仓**；正文未列 Hugging Face / 训练推理入口 |
| 数据集 | **未公开**（in-house 遥操作 / UMI / 第一人称视频 / 仿真；宣称每 1 美元采集配 3 美元质检） |
| 训练配方 | **本篇明确推迟**：「后续博文再讲如何训练 S1」 |
| 前序 LocoFormer | 论文 arXiv:2509.23745；官方实现未开源。社区有 [lucidrains/locoformer](https://github.com/lucidrains/locoformer) 非官方 WIP，**不能**当作 Skild 官方代码 |
| 可信度边界 | 产业官方博客，非 peer-reviewed；成功率、小时数、时间线为作者自报 |

## 核心摘录（归纳，非全文）

### 主张与定位

- 机器人仍处「BERT 时代」：新任务仍要数小时～数百小时部署域数据 + 微调；后训练数据够密时，**从零训练甚至能追上后训练基础模型**（引 Oh et al. 2026, FACTR 2）。预训练的主要（乃至唯一）目的应是 **in-context learning**。
- 任务用 **视频示范** 指定，不用语言。预训练强迫模型从示范读意图、功能对应与进度。
- 自评相对 concurrent ICL（[Generalist AI 2026](https://generalistai.com/blog/gen-1.5)、Jiang et al. 2026 RoboTTT）：后者多为 **短程或预训练已见**；S1 宣称首次在 **未见 + 最长约 10 分钟** 上展示 ICL。
- 本库 taxonomy 注意：RoboTTT 在 wiki 中归 **TTT（写权重）** 而非真 ICL；S1 文将其并列为 concurrent ICL，引用时需分机制。

### 训练叙事（高阶，细节未公开）

- 预训练 episodic 数据：**任务仅通过 in-context 示范指定**；示范可来自不同场景、视角、本体 → 策略必须学意图与对应。
- 元学习读法：预训练 = 外环「学会从上下文学习」；推理时示范驱动内环、**权重不变**。
- 算力底座：NVIDIA AI infrastructure。
- **一套权重** 覆盖文中全部演示；无微调、无后训练。

### 数据引擎三角

| 来源 | 硬件贴近 | 多样性 | 可扩展性 |
|------|----------|--------|----------|
| 机器人遥操作 | 高 | 低 | 低 |
| UMI | 中 | 中 | 中 |
| 第一人称人视频 | 低 | 高 | 高 |
| 仿真 | 中 | 低 | 高 |

主张：没有任何单一来源同时赢三轴，必须全开。质检投入约为采集的 **3×**。

### 评测数字（内部基准，自报）

对照：同一数据 / 架构（除 prompt embedding）/ 算力，ICL vs 语言条件 VLA，预训练 **1k–100k 小时**。指标为逐步成功率；失败用人干预恢复以便给每步打分。

| 设定 | 语言 VLA | ICL（S1 配方） |
|------|----------|----------------|
| 已见任务 @ 1k h | **53%** | 43%（小数据语言更稳） |
| 已见任务 @ 更大规模 | 被 ICL 反超（语言有歧义，示范锁模态） | 胜 |
| 未见任务 @ 100k h | **9%** | **66%**（约 **7×**） |
| 单次示范 ≈ 后训练条数 | — | 约 **380** 条后训练 episode 才追上单次 ICL |
| 未见任务 2000 条后训练 | **86%** | ICL 仍 66%（后训练最终更高，作者预期随 ICL 预训练缩小） |

未见任务定性示例：盆栽、摊饼、手冲咖啡、套件装配；单条 egocentric 人视频驱动。盆栽时间线：物料到办公室 → 布置场景 → 录示范 → **11 分钟后** 真机自治执行。

### 涌现行为（定性）

- 扰动鲁棒：滑动/替换物体、改光照（prompt 未展示）。
- 错误恢复：未见任务（如组装滑板轮）也会重试。
- 常识替换：示范用喷壶浇水但现场只有杯子 → 用杯子；杯子已满则只补一点。
- 示范纠错：把示范当 **目标规格** 而非轨迹复刻（示范失手打蛋，策略仍受控完成）。

### 分布偏移两轴（L1–L5）

- L1 同训练/同示范布置；L2 15 cm / 30°；L3 30 cm / 45°；L4 同 affordance 换物 + 竖直位移；L5 半数动作必须换臂。
- 相对训练条件：L5 下语言 VLA 退化可达 ICL 的 **3×**。
- 相对示范：ICL 对位姿/换物（L4）较稳，L5 换臂计划不同才明显掉点。

### 时间线（文内 Fig. 8）

| 节点 | 时间 |
|------|------|
| LocoFormer（运动 ICL，上下文累积在线经验） | 2025-09 |
| 操作域 in-distribution ICL 初现 | 2026-02 |
| S1 第一次摊饼（OOD） | 2026-05 |
| 本篇发布 | 2026-08 |

## 对 wiki 的映射

- [skild-s1](../../wiki/entities/skild-s1.md) — 本篇升格实体页
- [skild-ai](../../wiki/entities/skild-ai.md) — 公司入口
- [robot-in-context-learning](../../wiki/concepts/robot-in-context-learning.md) — 显式 ICL 预训练 + 长程未见任务产业样本
- [generalist-gen15-one-shot](../../wiki/entities/generalist-gen15-one-shot.md) — 短程涌现 ICL vs 长程显式 ICL 对照
- [paper-host-one-shot-human-video](../../wiki/entities/paper-host-one-shot-human-video.md) — 开源短程单视频对照（2026-09-04）
- [foundation-policy](../../wiki/concepts/foundation-policy.md) — 预训练目的 = ICL 的产业命题
- [embodied-scaling-laws](../../wiki/concepts/embodied-scaling-laws.md) — ICL vs 语言 prompt 的 scaling 分叉
- [data-flywheel](../../wiki/concepts/data-flywheel.md) — 分钟级部署回流预训练

## 可信度与使用边界

- **官方营销 + 技术叙事博客**；100k 小时、66%/9%、380 episode 交叉点均为内部基准插值。
- **架构、损失、上下文编码、机器人形态未公开**；不可当作可复现方法论文。
- 「首次 10 分钟未见 ICL」是作者立场；对照 GEN-1.5（3–12 s）与 RoboTTT（TTT，非 ICL）时按机制分栏，勿混排座次。

## Citation

```bibtex
@article{skild2026s1,
  author = {Skild AI},
  title  = {Introducing S1: In-Context Learning for Robotics},
  year   = {2026},
  month  = {August},
  url    = {https://skild.ai/blogs/s1},
}
```
