# Towards Machines with a Thousand Hands（Generalist AI）

> 来源归档（blog / Generalist AI 官方）

- **标题：** Towards Machines with a Thousand Hands
- **类型：** blog
- **作者 / 组织：** Generalist Team / Generalist AI
- **原始链接：** <https://generalistai.com/blog/towards-machines-with-a-thousand-hands>
- **发表日期：** 2026-07（博客 Citation 写 Jul 2026）
- **入库日期：** 2026-07-24
- **抓取方式：** Jina Reader（`r.jina.ai`）；官方页直连 HTML 仅作交叉核对
- **一句话说明：** Generalist **GEN-1** 将同一具身基础模型扩展到 **约 9000 种末端执行器变体**（五指手、螺丝刀、夹钳、打蛋器、刮铲等）与 **>50 万小时** 真实交互预训练数据；主张「手可变、物理不变」，用多手预训练学通用 sensorimotor / physical commonsense representations，并报告任务向量式权重更新分析（相对范数 **2.5%–11.4%**）与 **任务中途换手** 的同模型在线适应演示。

## 开源 / 项目页核查（步骤 2.5）

| 项 | 结论（截至 2026-07-24） |
|----|-------------------------|
| 本篇博客 / GEN-1 项目页 | **无**独立 `*.github.io` 式研究项目页；入口为公司博客 |
| 代码 / 权重 | **确认未开源**（公司站与 GEN-1 / 本篇正文未见 GitHub、Hugging Face 等公开训练/推理入口） |
| 数据集 | **未公开**（宣称 in-house 半百万小时级交互数据；预训练侧重人类可穿戴交互，后训练用少量机器人数据） |
| 可信度边界 | 产业官方博客，非 peer-reviewed；定量多为自报演示与内部分析 |

## 核心摘录（归纳，非全文）

### 主张与定位

- **GEN-1**（前序：[GEN-1 博文](https://generalistai.com/blog/gen-1)）作为最新具身基础模型，现支持广泛 **robot end effectors**：从五指拟人手，到带新驱动模式的专用工具，再到标准两指夹爪的改装。
- 目标叙事：单一 base model 学习可跨「截然不同物理交互方式」迁移的 sensorimotor policies。
- 类比：**多语言预训练**（Conneau et al. 2020）与 **multilingual CoT**（Shi et al. 2022）——每只手是一套物理交互「词汇」；跨手学习有助于分离「工具特异」与「世界共通」知识。

### 数据与形态覆盖

| 维度 | 博客要点 |
|------|----------|
| 预训练数据 | 公司 in-house robotics 数据集；现含 **> half a million hours** 真实交互 |
| 末端变体 | 约 **9,000** 种变体（含 off-the-shelf、打印件、对标准两指夹爪的定制改装） |
| 设计动机 | 暴露模型于广泛 **接触物理**（几何、摩擦、力、动力学）；部分形态受商业用例启发 |
| 示范工具族 | 超宽薄指搬箱、动力螺丝刀拧螺栓、夹钳放蛋、打蛋器搅拌、刮铲/清扫工具等 |

### 核心论点：「Hands can change, physics does not」

- 每个末端是不同的 **sensorimotor interface**；跨数千接口预训练 → 通用 **physical commonsense**（见姊妹文 [Physical Commonsense](https://generalistai.com/blog/physical-commonsense)）。
- 并非每只手都等权有用：部分末端学习信号弱；两指夹爪等「主流手」可能像英语在多语言模型中一样占主导——团队称在刻意扩展数据并在真实基准上评估。

### 如何度量「新手」新颖度（task vector 视角）

- 将微调后与预训练权重差视为 **task update / task vector**（Ilharco et al. 2023）。
- 相对参数范数变化约 **2.5%–11.4%**；**动力螺丝刀** 比 **夹钳** 需要更多「再教育」。
- 分解到架构子系统（传感处理、harmonic reasoning、驱动）：例如 **打蛋器** 比削皮器更大幅移动 **sensor-processing** 权重（细丝几何难感知）→ 指向数据干预（多采细、视觉稀疏工具）。

### 任务中途换手（on-the-fly）

- 演示：rollout 中途物理更换末端，**同一模型继续运行**——感知新工具、条件化所见形态，改换轨迹与接触策略以达成同一目标。
- 机制解读：混合多手数据迫使模型 **按眼前手条件化行为**，而非死记固定操纵套路；形成形状–接触–驱动策略的先验。

### 形态学展望（Cambrian explosion）

- 自然未收敛于单一操纵方案（喙、象鼻、吸盘、花粉篮等）。
- 机器人可通过 **tool changer** 换手、组合驱动模式；五指手应是工具箱中的一种，而非唯一目标。
- 结语：通用物理智能理解交互底层物理后，手的形状退居次要——吸盘、夹爪、刷子、焊枪皆为同一智能的接口。

## 对 wiki 的映射

- [generalist-gen1-thousand-hands](../../wiki/entities/generalist-gen1-thousand-hands.md) — 本篇升格实体页
- [generalist-ai-robotics](../../wiki/entities/generalist-ai-robotics.md) — 公司入口页更新
- [topic-cross-embodiment](../../wiki/overview/topic-cross-embodiment.md) — 跨末端 / 跨接口作为跨具身特例
- [foundation-policy](../../wiki/concepts/foundation-policy.md) — 商业通才策略对照
- [embodied-scaling-laws](../../wiki/concepts/embodied-scaling-laws.md) — 多接口数据多样性 scaling
- [manipulation](../../wiki/tasks/manipulation.md) — 多工具 / 多末端操作

## 可信度与使用边界

- **官方营销 + 技术叙事博客**，非独立评测；成功率、小时数、变体数为作者自报。
- **权重、训练配方、完整基准表未公开**；不可当作可复现方法论文使用。
- 「千手 / Cambrian」为愿景修辞；工程读者应聚焦 **多末端混合预训练 + 视觉条件化 + 权重更新诊断数据管线** 三条可迁移思想。

## Citation

```bibtex
@article{generalist2026thousandhands,
  author  = {Generalist Team},
  title   = {Towards Machines with a Thousand Hands},
  journal = {Generalist AI Blog},
  year    = {2026},
  note    = {https://generalistai.com/blog/towards-machines-with-a-thousand-hands},
}
```

## 姊妹入口（未全文 ingest）

- [GEN-1: Scaling Embodied Foundation Models to Mastery](https://generalistai.com/blog/gen-1)
- [GEN-0](https://generalistai.com/blog/gen-0)
- [Physical Commonsense](https://generalistai.com/blog/physical-commonsense)
- Postscript 推荐：[The Beetle Roboticists: A Parable](https://mtmason.com/the-beetle-roboticists-a-parable/)（Mason, 2024）
