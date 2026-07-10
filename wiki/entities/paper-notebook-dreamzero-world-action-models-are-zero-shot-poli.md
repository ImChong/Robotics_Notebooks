---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2602.15922"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_dreamzero-world-action-models-are-zero-shot-poli.md
summary: "DreamZero 提出 World Action Model (WAM)：在预训练视频扩散骨干上联合去噪未来视频与动作，把动作学习从「状态-动作模仿」转成「对齐预测视觉未来」的逆动力学；因此能从异构、非重复机器人数据学到通才策略，在未见任务/环境上零样本泛化比 SOTA VLA >2×，并通过 DreamZero-Flash + 系统优化把 14B 扩散模型压到 7 Hz 真机闭环。"
---

# DreamZero

**DreamZero: World Action Models are Zero-shot Policies** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

DreamZero 提出 World Action Model (WAM)：在预训练视频扩散骨干上联合去噪未来视频与动作，把动作学习从「状态-动作模仿」转成「对齐预测视觉未来」的逆动力学；因此能从异构、非重复机器人数据学到通才策略，在未见任务/环境上零样本泛化比 SOTA VLA >2×，并通过 DreamZero-Flash + 系统优化把 14B 扩散模型压到 7 Hz 真机闭环。

## 英文缩写速查

| 缩写 | 全称 | 解释 |
|---|---|---|
| WAM | World Action Model | 世界-动作模型：联合预测未来世界状态（视频）与动作 |
| VLA | Vision-Language-Action | 视觉-语言-动作模型，多从静态 VLM 初始化 |
| IDM | Inverse Dynamics Model | 逆动力学：由未来观测反推动作 |
| DiT | Diffusion Transformer | 扩散 Transformer 主干 |
| CFG | Classifier-Free Guidance | 无分类器引导，需条件/无条件两次前向 |
| KV-cache | Key-Value Cache | 自回归推理缓存，避免重复计算历史 token |

## 为什么重要

- **路线图顶点**：从 PPO → 模仿 → 扩散 → 世界模型，WAM 是「世界模型直接当策略」的落地形态
- **GR00T 2 技术源头**：NVIDIA 称 GR00T 2 基于 DreamZero 研究，MolmoSpaces / RoboArena 榜单领先
- **数据经济学**：不必为每个新动作采集重复 teleop；异构日常轨迹 + 视频先验即可
- **跨本体**：人类/异构机器人**仅视频**即可给目标机注入新技能先验

## 解决什么问题

SOTA **VLA** 擅长语义泛化（换物体、换语言指令），但在**新环境**和**新物理动作/技能**上泛化弱：

## 核心机制

1. **WAM 范式**：联合预测视频+动作，把「改进机器人能力」归结为「改进视频生成 + 对齐」。
2. **零样本泛化**：未见动词/动作/环境上，真机任务进度平均 **>2× SOTA VLA**；任务特定后训练后环境泛化仍 **+10%**。
3. **异构数据有效性**：~500 h 非重复真实轨迹即可训出通才策略，打破「每任务需大量重复 demo」惯例。
4. **跨本体迁移**：仅视频示范（人类 12 min / 异构机器人 20 min）→ 未见任务 **+42%** 相对提升；30 min play data 可 **few-shot 适配新 embodiment** 并保留零样本能力。
5. **38× 推理加速**：14B 扩散 WAM 首次达到 **7 Hz 真机闭环**；开源权重、推理与 RoboArena / PolaRiS / Genie Sim 3.0 评测代码。

方法拆解（深读笔记小节）：问题形式化：联合分布 = 视频预测 × 逆动力学；模型架构（Figure 4）；Flow-Matching 训练目标；闭环推理：KV-cache + 真值观测回填；实时执行：从 5.7 s/chunk 到 7 Hz；数据与泛化洞见。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/DreamZero_World_Action_Models_are_Zero-shot_Policies/DreamZero_World_Action_Models_are_Zero-shot_Policies.html> |
| arXiv | <https://arxiv.org/abs/2602.15922> |
| 机构 | NVIDIA（合作含 UC Berkeley、CMU 等） |
| 发表 | 2026-02-17 (arXiv) |
| 项目主页 | [dreamzero0.github.io](https://dreamzero0.github.io/) |
| 源码 | [dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero) |
| 笔记阅读日期 | 2026-06-08 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_dreamzero-world-action-models-are-zero-shot-poli.md](../../sources/papers/humanoid_pnb_dreamzero-world-action-models-are-zero-shot-poli.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/DreamZero_World_Action_Models_are_Zero-shot_Policies/DreamZero_World_Action_Models_are_Zero-shot_Policies.html>
- 论文：<https://arxiv.org/abs/2602.15922>

## 推荐继续阅读

- [机器人论文阅读笔记：DreamZero](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/DreamZero_World_Action_Models_are_Zero-shot_Policies/DreamZero_World_Action_Models_are_Zero-shot_Policies.html)
