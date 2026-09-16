# 具身前沿 | Embodied In-Context Learning（ICL）调研综述：从 Few-Shot Imitation 到 Physical Prompting

> 来源归档（blog / 微信公众号）

- **标题：** 具身前沿 | Embodied In-Context Learning（ICL）调研综述：从 Few-Shot Imitation 到 Physical Prompting
- **类型：** blog
- **作者：** 杰西 / Mbot具身智能实验室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/WmtTSKS85i2ZJJijFewy3A
- **发表日期：** 2026-09（文内标注「2026 年 9 月调研」）
- **入库日期：** 2026-09-16
- **抓取方式：** WebFetch（`mp.weixin.qq.com`；本环境未预装 `wechat-article-for-ai`）
- **原始抓取落盘：** [`sources/raw/wechat_mbot_embodied_icl_survey_2026-09.md`](../raw/wechat_mbot_embodied_icl_survey_2026-09.md)
- **一句话说明：** 以 **VIMA→KAT→ICWM→LocoFormer→RoboTTT/WAM-TTT→GEN-1.5/S1** 八项工作为主线的具身 ICL 技术演化综述；提出 **Pure ICL vs TTT** 双层口径与 **Task / Embodiment / History** 三类学习对象；覆盖机构版图与未来评测方向。

## 核心摘录（归纳，非全文）

### 部署范式转变

- **传统 VLA：** 新任务 → 采集 demo → Fine-tuning/Post-training → 部署新策略。
- **ICL 目标：** 新任务/环境 → Demo / Human Video / Interaction History → Context/Memory → 基础模型 **即时适应**（零梯度或轻量 TTT），无需重训主权重。

### 双层口径（文内统一框架）

| 范式 | 测试时参数更新 | 典型机制 | 代表 |
|------|----------------|----------|------|
| **Pure ICL** | 否 | Attention、KV Cache、Transformer-XL、上下文激活 | KAT、ICWM、LocoFormer、GEN-1.5、S1 |
| **TTT / Fast-Weight** | 是（仅快权重） | Fast Weights、Adaptive Memory、自监督内循环 | RoboTTT、WAM-TTT |

### ICL「学」什么（三类对象）

| 类型 | Context 来源 | 归纳对象 |
|------|--------------|----------|
| **Task / Behavior** | 机器人示范、人视频、XR 遥操作 | 要做什么、怎么做 |
| **Embodiment / World** | 任务无关主动探索 | 相机、运动学、动力学、embodiment |
| **History / Memory** | 自身 rollout、失败、跨 episode 历史 | 进度、恢复、长期适应 |

### 技术演化三阶段（文内 §8）

1. **Prompt 化（2023–2024）：** VIMA 统一多模态 prompt；KAT 证明冻结 LLM 可做 few-shot ICIL。
2. **在线适应化（2025–2026）：** ICWM 系统辨识；LocoFormer 跨 episode 长上下文；RoboTTT/WAM-TTT 参数化测试时记忆。
3. **Foundation-Scale ICL（2026–）：** GEN-1.5 涌现 physical prompting；S1 显式 ICL 预训练 + 长时程未见任务。

### 八项重点工作（文内 §4–5 深读摘要）

| 工作 | Context 来源 | 学习对象 | 参数更新 | 文内定位 |
|------|--------------|----------|----------|----------|
| VIMA (2023) | 文本+图像+视频 prompt | 任务指定 | 否 | 多模态 prompt 统一接口；L4 零样本 2.9× |
| KAT (2024) | ≤10 视觉-动作 demo | 行为模式 | 否 | 关键点 token + 冻结 GPT-4；>40 demo 触顶 |
| ICWM (2026) | N=5 任务无关探测 | 相机/构型/系统 | 否 | 测试时系统辨识；假 context 实验证真 ICL |
| LocoFormer (2025) | 跨 episode rollout | 形态/动力学 | 否 | TXL 长记忆；零样本 0.96→少样本 0.98 |
| RoboTTT (2026) | 人视频+失败+DAgger | 长时恢复/模仿 | 是（快权重） | 8K context；GDN 对照证梯度更新必要 |
| WAM-TTT (2026) | 无标注人玩耍视频 | 任务变体 steering | 是（video 侧） | WAM-ICL 7.1% vs TTT 46.2% |
| GEN-1.5 (2026) | 3–12s sensorimotor demo | 新任务/组合 | 否 | 涌现 one-shot ~59%；闭源 |
| S1 (2026) | 单条任务视频 | 未见+长时程 | 否 | 100k h 未见 66% vs 语言 VLA 9% |

### 核心矛盾（文内 §4.2）

1. Context **来自哪里**（demo / 人视频 / 自身探索 / 历史失败）？
2. Context **如何表示**（原始 token / 关键点 / 结构化计划 / latent）？
3. Context **存在哪里**（Attention/KV / TXL recurrent / Fast Weights）？
4. 如何 **证明真在用 Context**（unseen task、假 context、跨 embodiment）？

### 未来方向（文内 §7 要点）

- Language → **Physical/Video Prompt** 双接口。
- Task + Embodiment + World **统一上下文推断**。
- **Context length** 成为机器人 FM 新 scaling 轴。
- Pure ICL 与 TTT **混合部署**（短期 prompt + 长期 fast weights + retrieval）。
- **Human video** 规模化；分钟级组合与自恢复；严格 ICL 评测协议。

## 对 wiki 的映射

- **主更新：** [robot-in-context-learning](../../wiki/concepts/robot-in-context-learning.md) — 补双层口径、演化谱系、八项深读对照与未来评测轴。
- **交叉引用（复用既有节点）：** [GEN-1.5](../../wiki/entities/generalist-gen15-one-shot.md)、[S1](../../wiki/entities/skild-s1.md)、[RoboTTT](../../wiki/entities/paper-robottt-test-time-training-vla-context.md)、[WAM-TTT](../../wiki/entities/paper-wam-ttt-human-video-test-time-steering.md)、[四路线对比](../../wiki/comparisons/wam-ttt-robottt-stellavla-zero-wam-embodied-icl.md)、[LocoFormer 待深读](../../wiki/entities/paper-notebook-locoformer-generalist-locomotion-via-long-contex.md)。
- **与既有综述关系：** [具身智能之心 ICL 综述（2026-08-25）](wechat_embodied_heart_robot_icl_gen15_survey_2026-08-25.md) 侧重 **三类不确定性 taxonomy** 与 26 篇索引；本篇侧重 **八项主线深读 + 机构版图 + 演化时间线**，互补不重复。

### 文内其余工作（本 ingest 仅索引）

ICRT、Instant Policy、RICL、BPP、MimicDroid、StellaVLA、Zero-WAM 等 — 见文内 §9 延伸阅读。
