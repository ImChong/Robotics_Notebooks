# pamor_arxiv_2608_28213

> 来源归档（ingest）

- **标题：** PAMoR: Parameterized Affective Motion Generation in Real Time for Humanoid Robots
- **类型：** paper
- **来源：** arXiv:2608.28213（2026 预印本）
- **作者：** Yan Pan, Lingfan Bao, Tianhu Peng, Chengxu Zhou
- **机构：** 伦敦大学学院（University College London）计算机科学系
- **入库日期：** 2026-09-01
- **最后更新：** 2026-09-01
- **项目页：** <https://arxiv.org/abs/2608.28213>（截至入库日无独立 lab 项目页）
- **一句话说明：** 在 Unitree G1 上把 **效价–唤醒（V-A）** 从机器人运动学闭式标定，并用 **可组合潜扩散**（文本动作先验 + 效价/唤醒先验）实时生成可独立编辑动作与情感风格的全身运动；感知实验 Top-1 **0.384**，接近人体表演 **0.44** 基线。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / §I）

- **链接：** <https://arxiv.org/abs/2608.28213>
- **痛点：** 社交场景中人形运动需同时传达 **做什么** 与 **怎么感受**；虚拟人侧多靠参考片段或情感词，难以 **定量参数化**；人形侧连续 V-A 全身实时生成仍不足。
- **PAMoR 主张：** (a) 从 **姿态扩张 + 运动能量** 在机器人正运动学上闭式计算 V-A，**无需人工情感标注**；(b) 三个扩散先验在共享 MVAE 潜空间分别建模文本动作、效价、唤醒，推理时 **classifier-free 组合**；(c) 在 **29-DoF Unitree G1** 上自回归实时 rollout，动作与情感均可在线编辑。
- **对 wiki 的映射：**
  - [PAMoR 实体](../../wiki/entities/paper-pamor.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
  - [TextOp 实体](../../wiki/entities/paper-loco-manip-161-022-textop.md)（文本驱动人形运动基线）

### 2) V-A 运动学标定（§III-B）

- **效价 \(v\)：** 臂展三角形周长 \(s_t\) + 根高度 \(b_t^z\) + 躯干俯仰 \(\theta_t\)，等权 z-score 求和后 winsorize 到 \([-1,1]\)。
- **唤醒 \(a\)：** 关键点速度幅值均值 + 加速度幅值均值，同样 z-score 与 winsorize。
- **对 wiki 的映射：**
  - [PAMoR 实体](../../wiki/entities/paper-pamor.md) §核心原理

### 3) 可组合潜扩散（§III-C）

- **MVAE：** 9 层 Transformer，\(H{=}2\) 历史 + \(F{=}8\) 未来帧 → 128 维潜 token；三先验各约 17.5M 参数，8 层 Transformer 去噪器。
- **组合：** \(\hat{z}_0 = \hat{z}_\tau^u + \sum_{i\in\{\tau,v,a\}} \lambda_i(\hat{z}_i^c - \hat{z}_i^u)\)，\((\lambda_\tau,\lambda_v,\lambda_a)=(5,2,2)\)；推理 \(T{=}10\) 步 DDPM，单 primitive **78 ms**（RTX 5090）。
- **执行：** 冻结 MVAE 解码 primitive → **SONIC** 全身跟踪器真机执行。
- **对 wiki 的映射：**
  - [Diffusion Motion Generation](../../wiki/methods/diffusion-motion-generation.md)

### 4) 数据与训练语料（§III-D）

- **来源：** BABEL-annotated AMASS（GMR 重定向至 G1）+ GVHMR 视频重建 + G1 遥操作；15 类日常动作；三段插值为链；仿真 SONIC 过滤不可跟踪片段。
- **规模：** 10,095 训练链 / 2,406 验证链（20 Hz）；每链 3 段 → 30,285 训练段。
- **对 wiki 的映射：**
  - [Unitree G1](../../wiki/entities/unitree-g1.md)

### 5) 实验数字（§IV）

| 对比 | 要点 |
|------|------|
| vs TextOp / ECHO | FID、R@1、MM-Dist、Diversity 与文本基线 **持平**；MultiModality **0.882**（TextOp 0.521，ECHO 0.787） |
| vs SMooDi（风格参考） | R@1 **0.166**（风格牺牲动作）；PAMoR 保持 ~0.79 级文本对齐 |
| V-A 可控性 | 命令–测得秩相关 **0.95**；\(s_v{=}1.10\)，\(s_a{=}0.79\)；轴间相关 <0.06 |
| 感知实验 | Top-1 **0.384**、Top-3 **0.845**、\(\kappa_w{=}0.688\)、自然度 **4.05/5**；TextOp 情感 prompt Top-1 仅 **0.125**（ chance 0.125） |
| 消融 | 单先验联合 \((\tau,v,a)\)：Action Acc. **0.478** → 可组合 **0.748**；潜空间 vs 运动空间：唤醒相关 **0.786→0.947**，jerk **157.8→92.7** |

- **对 wiki 的映射：**
  - [PAMoR 实体](../../wiki/entities/paper-pamor.md) §实验与评测

### 6) 开源核查（步骤 2.5，2026-09-01）

| 项 | 状态 |
|----|------|
| arXiv 摘要页 / HTML | <https://arxiv.org/abs/2608.28213> — 方法图、用户研究与真机演示描述齐全 |
| 独立项目页 | **无** — 作者 UCL 主页与 arXiv 均未链出专用 `*.github.io` 或 lab 子页 |
| Code / GitHub | **未列出** — arXiv「Code, Data, Media」区无官方仓库；公开检索未见 PAMoR 官方实现 |
| 论文 Code availability | 正文未承诺「将开源」 |
| 结论 | **未开源** — 截至入库日无可辨识训练/推理/部署入口；源码运行时序图不适用 |

## 其他公开资料

- arXiv HTML：<https://arxiv.org/html/2608.28213>
- 作者通讯：chengxu.zhou@ucl.ac.uk
- 同实验室相关工作：[HIAER（arXiv:2506.01563）](../../sources/papers/humanoid_pnb_hierarchical-intention-aware-expressive-motion-g.md) — 场景 VLM 推断 V-A 驱动手势，非本文生成器条件化路线

## 对 wiki 的映射

- [paper-pamor.md](../../wiki/entities/paper-pamor.md)
- [pamor-arxiv.md](../sites/pamor-arxiv.md)
