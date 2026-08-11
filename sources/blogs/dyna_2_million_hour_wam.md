# Dyna-2: A 1-Million-Hour Scaling Law for World-Action Models

> 来源归档（blog / research post · Dyna Robotics 官方）

- **标题：** Dyna-2: A 1-Million-Hour Scaling Law for World-Action Models
- **类型：** blog / research technical report（公司站，非 arXiv）
- **作者 / 组织：** Dyna Robotics
- **原始链接：** <https://www.dyna.co/dyna-2>
- **发表日期：** 2026-08（页面 Date: August 2026；公关稿约 2026-08-10）
- **入库日期：** 2026-08-11
- **抓取方式：** 官方页 WebFetch（`www.dyna.co/dyna-2`）
- **一句话说明：** **Dyna-2** 是 Dyna Robotics 旗舰 **World-Action Model（WAM）**：在 **≥100 万小时** egocentric 人类操作视频上预训练（预训练 **不含** 机器人数据），报告 held-out 人数据幂律缩放，并主张首次测到 **人→机器人跨具身零样本** 的缩放律；后训练少量机器人数据即可上双臂 / 灵巧手 / 半人形原型。

## 开源 / 项目页核查（步骤 2.5）

| 项 | 结论（截至 2026-08-11） |
|----|-------------------------|
| 研究入口 | <https://www.dyna.co/dyna-2>（公司 Research 长文；另有公司首页 <https://www.dyna.co/>） |
| 独立 `*.github.io` 项目页 | **无** |
| 代码 / 权重 | **确认未开源** — 研究页与公司首页未见 GitHub / Hugging Face / Zenodo 训练或推理入口 |
| 数据集 | **未公开**（宣称自建 + 数据伙伴的百万小时级 egocentric 语料；伪动作来自 3D 手姿） |
| arXiv / peer-review | **未见** 同名 arXiv；正文以公司 technical report + BibTeX `dyna2026dyna2` 自引 |
| 可信度边界 | 产业官方研究博文；定量多为内部 ladder / 盲测 / 客户现场 pass criteria，待独立复现 |

## 核心摘录（归纳，非全文）

### 主张（回答三个缩放问题）

1. **预训练数据源应是什么？** — 经济上有价值的任务已由人类完成 → 主源应为 **传感化人类视频**（而非仅靠遥操作机端数据）。
2. **缩放该源是否对机器人有效？** — 报告 **人→机零样本** 离线指标随人数据小时数单调改善（39 任务 × 两套 YAM 双臂）。
3. **何种建模/目标才能让缩放律出现？** — **世界建模（视频预测）+ 视频共训** 是跨具身缩放出现的必要条件；纯 action-only 在同架构下不出现可靠缩放。

### 架构与目标（§2）

| 要素 | 要点 |
|------|------|
| 骨架 | **Mixture-of-Transformers**；各模态独立 tokenize；视频 / 动作各有 **DiT** 层，可互注意力 |
| 本体觉 | proprio  tokenize 后直接进 **action transformer** |
| 掩码 | 视频 token **因果**；动作 token **双向**，并 attend 观测上下文视频；文本经 cross-attn 进视频侧（**不**直接进动作） |
| 深度选择 | 早期层保留多数时序推理 → **动作塔刻意更浅**，在早期层接入视频流，换实时延迟 |
| 训练 | **Flow matching** 对未来视频 latent \(z\) 与动作 chunk \(a\) 分别拟合速度场 |
| 缩放律实验用目标 | **Co-training（边际）**：视频损失与动作损失共享 trunk，但 \(u^{\mathrm{act}}\) **不**以 \(z_t\) 为输入 → 推理保持 **reactive**（不生成、不 attend 预测未来视频） |

### 数据梯子（§3）

- 语料：**>1,000,000 h** 头戴 egocentric 日常操作（烹饪、整理、折叠、装配等）；清洗 + 手姿提取；通过质量门的片段给出 **3D 手姿** → 伪动作（腕轨迹 + 拇指–食指孔径 grasp 信号）。
- **嵌套精确小时子集：** 1k / 10k / 100k / 1M h，各源比例固定；另固定 **100 h** held-out 人验证集。
- **不做** 视觉/运动学具身对齐预处理——刻意只研究 **缩放本身**。
- 机器人离线评测：**39 任务**（内部 12 + [xdof ABC](https://arxiv.org/abs/2606.27375) 27），两套静止 **YAM** 双臂；预训练 **零** 机器人轨迹。

### 缩放律结果（摘要）

| 轴 | 指标行为（博客自报） |
|----|----------------------|
| Held-out 人 | MSE / L1 / acc@0.1 / acc@0.5 均幂律改善；例：MSE \(= 0.0691\cdot D^{-0.0184}\)，\(R^2=0.919\) |
| Zero-shot 机（离线） | 同组检查点在 39 任务上 MSE↓、acc@0.5↑ 随人小时单调；约 **10k→100k** 出现转折 |
| 后训练真机（14 任务） | 仅机器人后训练（每任务 ≤10 h；无对齐/共训）；归一化均值 **20% → 28% → 45% → 53%**（1k→1M） |
| 极端数据效率 | **Bottle Cap Untwisting**（约 **10–13 min** 遥操作）随预训练从 ~10% 升至 ~50%；**Lockbox Key Turning** 在 ≤100k 为 0%，1M 达 **90%** |

### 消融：世界建模与「视频新缩放轴」

- 固定手姿标注量，比较 **action-only / joint / video co-train**：joint 在 39/39 任务上优于 action-only；**仅 video co-train** 随 action 数据增长而继续改善。
- 固定 action-labelled 小时（50k 或 250k），只缩放 **无动作标签视频**：零样本机器人 MSE 单调下降 → 主张 **视频是新缩放轴**；同设定对人 held-out 几乎无益甚至略差。

### 额外能力（§4，生产配方检查点）

| 能力 | 要点 |
|------|------|
| WAM vs VLA（早版 Dyna-2 vs Dyna-1） | 匹配数据/步数下 WAM 成功率约 **1.55×** VLA；head-to-head 赢 **65%**（作者称对 WAM 不公平：早版无 1M、且曾用 action-only） |
| 生产现场零样本 | 同后训练预算：客户现场 pass **Dyna-1 46% → Dyna-2 87%**（室内皆近 100%） |
| 语言跟随 | action-only 0.35 → video co-train early 0.67 → full corpus **0.96**（四任务语言反事实协议） |
| 一步视频蒸馏 | 控制式学生–演化目标；3s 三视角操纵视频：**10,203 ms → 110 ms**（1×H100），相对 teacher 约 **93×** |

### 产品栈语境（公司首页交叉）

公司站将 **DYNA-2** 定位为 System 1（中层灵巧），上接 **DYNA-VLM**（System 2），下接 **DYNA-System0** 全身控制与 **DYNA-SAUR** 传感/本体；部署叙事覆盖洗衣、餐饮、工厂装配。

## BibTeX（页面提供）

```bibtex
@article{dyna2026dyna2,
  author = {{Dyna Robotics}},
  title  = {Dyna-2: A 1-Million-Hour Scaling Law for World-Action Models},
  year   = {2026},
  month  = {August},
  url    = {https://dyna.co/dyna-2},
}
```

## 对 wiki 的映射

- 沉淀实体：**[`wiki/entities/dyna-2.md`](../../wiki/entities/dyna-2.md)**
- 概念交叉：[World Action Models](../../wiki/concepts/world-action-models.md)、[Embodied Scaling Laws](../../wiki/concepts/embodied-scaling-laws.md)
- 方法对照：[EgoScale](../../wiki/methods/egoscale.md)（~20k h 人视频 VLA + 对齐 mid-training）、[VLA](../../wiki/methods/vla.md)
- 项目页归档：[sources/sites/dyna-co-dyna-2.md](../sites/dyna-co-dyna-2.md)、公司站 [sources/sites/dyna-co.md](../sites/dyna-co.md)
