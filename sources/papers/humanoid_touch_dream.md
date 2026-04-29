# Humanoid Touch Dream

> 来源归档（ingest）

- **标题：** Learning Versatile Humanoid Manipulation with Touch Dreaming
- **类型：** paper
- **来源：** arXiv:2604.13015v2 / PDF / GitHub project page
- **入库日期：** 2026-04-29
- **最后更新：** 2026-04-29
- **一句话说明：** 提出 Humanoid Transformer with Touch Dreaming (HTD)，把触觉作为人形机器人接触丰富型移动操作策略的核心模态，并用未来手部力与触觉 latent 预测作为行为克隆的辅助目标。

## 原始资料入口

- **arXiv HTML：** <https://arxiv.org/html/2604.13015v2>
- **arXiv PDF：** <https://arxiv.org/pdf/2604.13015>
- **GitHub：** <https://github.com/chrisyrniu/humanoid-touch-dream>
- **项目主页：** <https://humanoid-touch-dream.github.io/>

> 备注：这三个入口对应同一篇论文/项目，因此本仓库只建立一个 source 节点。GitHub 仓库在 2026-04-29 检查时仅包含论文 README、MIT License，并声明 whole-body controller、teleoperation 与 HTD policy learning 代码计划在 2026 年 5 月初发布。

## 核心论文摘录（MVP）

### 1. HTD：用 Touch Dreaming 正则化人形操作策略（Niu et al., 2026）

- **链接：** <https://arxiv.org/abs/2604.13015>
- **核心贡献：** HTD 是一个多模态 encoder-decoder Transformer，把多视角视觉、本体感受、手部关节力和双手分布式触觉输入编码到共享 latent，再通过 action experts 输出 torso、end-effector、velocity 与 dexterous-hand action chunks。
- **Touch Dreaming：** 除了行为克隆动作预测，训练时额外预测未来手部关节力和未来触觉 latent。触觉 latent 目标由 EMA tactile tokenizer 提供，避免单独触觉预训练，也避免直接回归稀疏高维 raw tactile array。
- **对 wiki 的映射：**
  - [HTD 方法页](../../wiki/methods/humanoid-transformer-touch-dreaming.md)
  - [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)
  - [触觉感知](../../wiki/concepts/tactile-sensing.md)

### 2. 分层系统：LBC + VR teleoperation + dexterous hands + tactile sensing

- **链接：** <https://arxiv.org/html/2604.13015v2>
- **核心贡献：** 论文不是只提出一个策略网络，而是搭建了完整的人形机器人移动操作系统：RL-based lower-body controller 负责稳定下肢/躯干，VR 遥操作映射上身和手部动作，IK 与 DexPilot-style retargeting 执行末端与灵巧手目标。
- **数据结构：** 遥操作数据同步记录头部/腕部多视角 RGB、本体感受、手部关节力、双手触觉读数和全身 action targets。每只手包含 1062 维触觉观测，覆盖 17 个空间感知区域。
- **对 wiki 的映射：**
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
  - [Teleoperation](../../wiki/tasks/teleoperation.md)
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)

### 3. 真实任务结果与消融结论

- **链接：** <https://arxiv.org/html/2604.13015v2>
- **核心贡献：** 在 Insert-T、Book Organization、Towel Folding、Cat Litter Scooping、Tea Serving 五个真实接触丰富型任务上评估；每个任务/方法 20 次 real-world trials。
- **关键结论：** HTD 相比更强 ACT baseline 的平均成功率提升 30.0 个百分点，约 90.9% relative improvement；相比 raw tactile prediction，latent tactile dreaming 在成功率上有约 30% relative gain。
- **工程含义：** 单纯把触觉作为输入不稳定，必须通过未来接触预测迫使共享 trunk 学到接触动态；部署时 dream heads 不参与控制，因此推理路径仍保持简洁。
- **对 wiki 的映射：**
  - [Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md)
  - [Behavior Cloning with Transformer](../../wiki/methods/bc-with-transformer.md)
  - [Action Chunking](../../wiki/methods/action-chunking.md)

## 当前提炼状态

- [x] 论文/HTML/PDF/GitHub 合并为单一 source 节点
- [x] 核心系统、方法与实验结论提炼
- [x] 升格为 HTD 方法页
- [ ] 后续在代码发布后补充实现结构、训练配置和可复现实验入口
