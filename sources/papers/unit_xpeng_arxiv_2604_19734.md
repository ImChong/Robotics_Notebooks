# UniT — Toward a Unified Physical Language for Human-to-Humanoid Policy Learning and World Modeling

> 来源归档（paper）

- **标题：** UniT: Toward a Unified Physical Language for Human-to-Humanoid Policy Learning and World Modeling
- **类型：** paper
- **机构：** 小鹏机器人（XPENG Robotics）；清华大学（Tsinghua University）；香港大学（The University of Hong Kong）
- **作者：** Boyu Chen；Yi Chen；Lu Qiu；Jerry Bai；Yuying Ge；Yixiao Ge
- **原文版本：** arXiv:2604.19734v2（2026-10-11；初版 2026-04-21）
- **核对范围：** [arXiv v2 HTML 原文](https://arxiv.org/html/2604.19734v2) §1–6，含公式（1）–（6）、表 1–3、跨本体消融及讨论；并与官方项目页及公开训练 README 核对复现范围
- **通读：** 已完成
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
- **摘录要点：** Vision / Action / Fusion 三分支经共享 RQ-VAE 量化，三种分支 token 都需视觉 FDM 与动作解码器交叉重构。下游区别：**VLA-UniT** 的融合分支离散 token 是 VLM 的交叉熵监督目标，连续动作仍由 flow-matching expert 生成；**WM-UniT** 则将**动作分支量化前连续特征**经 MLP/cross-attention 注入 Cosmos Predict 2.5。论文全量 RoboCasa GR1 **66.7%**（比 FLARE +11.7、同架构 GR00T +18.9 个百分点），WM 支持人动作条件下的人形视频生成。
- **对 wiki 的映射：**
  - [WAM×运动控制五路径](../../wiki/overview/wam-motion-control-five-paths.md) — ⑤ 动作表示入口
  - [VLA](../../wiki/methods/vla.md)、[World Action Models](../../wiki/concepts/world-action-models.md)

### 3) 开源状态

- **链接：** <https://github.com/xpeng-robotics/UniT>
- **摘录要点：** **部分功能公开**（Apache-2.0）：可运行 VLA-UniT tokenizer、dual-system 训练、RoboCasa 在线/离线评测脚本和 HF 策略权重；仓库 README 未列出 WM-UniT/Cosmos 独立运行入口。GR1-full 与 EgoDex+GR1-100 公开配方的参考成功率分别为 **66.4%** 与 **50.9%**，不同于论文图示数值；运行配方的显存假设为每卡 120+GB（仿真 rollout 可在 24GB GPU）。
- **对 wiki 的映射：**
  - [xpeng-robotics/UniT 仓库归档](../repos/xpeng_robotics_unit.md)

### 4) 评测边界：真实执行与视频生成不可混用

- **链接：** <https://arxiv.org/html/2604.19734v2>
- **摘录要点：** 论文 few-shot 从 45.5% 到人机混训的 50.0%；IRON 真机未见堆叠任务通过人机混训达到 60%，机器人-only UniT 为 10%，GR00T 为 0%；世界模型跨本体条件的 1–5 分一致性是 Gemini-3-Pro 自动评审，并非真实机器人成功率。
- **对 wiki 的映射：**
  - [UniT 实体页](../../wiki/entities/paper-unit-unified-physical-language.md)
  - [跨本体迁移纵深路线](../../roadmap/depth-cross-embodiment.md)

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
