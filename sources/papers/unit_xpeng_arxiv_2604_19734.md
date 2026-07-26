# UniT — Toward a Unified Physical Language for Human-to-Humanoid Policy Learning and World Modeling

> 来源归档（paper）

- **标题：** UniT: Toward a Unified Physical Language for Human-to-Humanoid Policy Learning and World Modeling
- **类型：** paper
- **机构：** XPENG Robotics；Tsinghua University；The University of Hong Kong
- **arXiv：** <https://arxiv.org/abs/2604.19734>
- **项目页：** <https://xpeng-robotics.github.io/unit/>
- **代码：** <https://github.com/xpeng-robotics/UniT>（Apache-2.0）
- **权重：** <https://huggingface.co/xpeng-robotics/VLA-UniT-checkpoints>
- **入库日期：** 2026-07-26
- **一句话说明：** 通过视觉锚定的三分支交叉重构，学习人与人形共享的统一离散运动分词（Unified Latent Action）；同时服务 **VLA-UniT 策略学习** 与 **WM-UniT 动作条件世界建模**。

## 核心论文摘录（MVP）

### 1) 问题：人体数据丰富，但跨本体鸿沟卡缩放

- **链接：** <https://arxiv.org/abs/2604.19734>
- **摘录要点：** 人形数据稀缺而 egocentric 人体数据丰富；传统 retargeting 为每台机器人定制运动学求解，且训练仍常把「人视觉」与「机动作」硬配对。UniT 主张用 **共享潜动作空间** 做介质：异构运动学共享一致视觉后果，以视觉为锚强制视觉↔动作交叉重构。
- **对 wiki 的映射：**
  - [UniT 实体页](../../wiki/entities/paper-unit-unified-physical-language.md)
  - [Motion Retargeting Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md)

### 2) 方法：Tri-branch + 共享 RQ-VAE codebook

- **链接：** <https://xpeng-robotics.github.io/unit/>
- **摘录要点：** Vision / Action / Fusion 三编码器 → 共享 RQ-VAE 码本 → 视觉与动作解码器从同一 token 交叉重构。下游：(1) **VLA-UniT** 预测统一 token，RoboCasa GR1 全数据成功率 **66.7%**（相对 FLARE +11.7pp，相对同架构 GR00T 基线 +18.9pp），并支持真机 IRON-R01-1.11 与 zero-shot 任务迁移；(2) **WM-UniT** 用统一 token 作动作条件，支持人动作→人形视频生成与跨本体动力学迁移。
- **对 wiki 的映射：**
  - [WAM×运动控制五路径](../../wiki/overview/wam-motion-control-five-paths.md) — ⑤ 动作表示入口
  - [VLA](../../wiki/methods/vla.md)、[World Action Models](../../wiki/concepts/world-action-models.md)

### 3) 开源状态

- **链接：** <https://github.com/xpeng-robotics/UniT>
- **摘录要点：** **已开源**（Apache-2.0）：仓库含 `gr00t`/`preprocessing`/`scripts`/`examples` 等；HF 提供 VLA-UniT checkpoints。后续 Fe₀ 在 UniT 统一物理语言上放大异构数据（见项目页 Fe₀ 博文）。
- **对 wiki 的映射：**
  - [xpeng-robotics/UniT 仓库归档](../repos/xpeng_robotics_unit.md)

## 关键术语

- **UniT：** Unified Latent Action Tokenizer via Visual Anchoring
- **VLA-UniT / WM-UniT：** 同一分词器分别接到策略与世界模型

## 关联 Wiki 页面

- [paper-unit-unified-physical-language](../../wiki/entities/paper-unit-unified-physical-language.md)
- [wam-motion-control-five-paths](../../wiki/overview/wam-motion-control-five-paths.md)

## 当前提炼状态

- [x] arXiv / 项目页 / 代码 / HF
- [x] 策略与 WM 双下游数字
- [x] wiki 映射
