# Green-VLA：分阶段 Vision–Language–Action 通才操作与人形部署（arXiv:2602.00919）

> 来源归档（ingest）

- **标题：** Green-VLA: Staged Vision–Language–Action Model for Generalist Robots Manipulation
- **类型：** paper / vla / multi-embodiment / humanoid / flow-matching / rl-post-training
- **arXiv abs：** <https://arxiv.org/abs/2602.00919>
- **arXiv HTML：** <https://arxiv.org/html/2602.00919>
- **PDF：** <https://arxiv.org/pdf/2602.00919>
- **项目页：** <https://greenvla.github.io/>
- **代码：** <https://github.com/greenvla/GreenVLA>
- **机构：** Sber Robotics Center
- **硬件：** **Green** 人形（上身 **32 DoF**：头、躯干、双臂、灵巧手）；评测另含 AgileX Cobot Magic、WidowX、Google Robot、CALVIN 等
- **规模：** 约 **5B** 参数（**Qwen3-VL-4B-Instruct** + flow-matching action expert；早期版 **PaliGemma 3B** 约 4B）；R0 在 **64×H100** 上 **>10⁵** 步
- **数据：** **24M** 非机器人多模态样本（L1）+ **>3000 h** 跨本体机器人演示（R0，含 AgiBotWorld、DROID、Galaxea、Green Humanoid 等）；自采 Green Humanoid **48 h** 经镜像/时间反转扩至 **167 h**
- **入库日期：** 2026-06-18
- **一句话说明：** **五阶段 VLA 课程**（L0 底座 VLM → L1 网页多模态接地 → R0 多本体预训练 → R1 本体 SFT → R2 RL 对齐）配合 **DataQA 质量过滤**、**64 维语义统一动作空间** 与 **flow-matching 动作专家**；在 **<π₀ 数据量** 下于 Simpler / ALOHA 清理桌等 R0 阶段即超多项基线，**R2** 在 WidowX 上较 R1 **+24%** 绝对成功率，并在 **Green 人形** 上零样本跨本体完成双手分拣、递送与长程清理。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://greenvla.github.io/> | 人形分拣/清理、电商货架 JPM、Simpler 与 CALVIN 曲线 |
| 代码 | <https://github.com/greenvla/GreenVLA> | 官方开源仓库（截至入库日已公开） |
| 对照 VLA | π₀ / π₀.₅、GR00T N1、AgiBot GO-1、WALL-OSS、EO-1、X-VLA | 论文 Table 2–4 主要基线 |
| 高层规划 | GigaChat / GigaVision VLM | 推理时冻结的任务分解与反馈模块 |

## 摘要级要点

- **问题：** 纯 **行为克隆（BC）** 在长程接触操作上快速饱和；异构数据集 **观测/动作/采样率** 不一、质量参差；朴素 **padding 统一动作** 引入虚假梯度、破坏跨本体迁移。
- **Green-VLA 回答：**
  - **数据：** DataQA 用抖动 $S_{\text{tremble}}$、清晰度 $S_{\text{sharp}}$、视觉多样性 $D_{\text{vis}}$、状态方差等筛段；**光流对齐** 重采样统一执行速度；**$\alpha_t$ 课程采样** 从均匀混合渐近目标权重，稳定多本体 flow matching。
  - **动作：** 固定语义槽位 $\mathcal{A}_u \subset \mathbb{R}^{64}$ + **embodiment/control-type prompt** $c_e$ + 掩码 BC；对目标人形做显式 **retargeting**。
  - **模型：** VLM 编码多视角 RGB + 本体 + 语言 → **flow-matching 动作专家**；辅助 **episode progress**、**GMM OOD 检测**、**JPM + ΠGDM 引导** 精确抓取点。
  - **部署：** 冻结 **GigaVision** 高层任务规划器分解子任务；Green 人形控制 **全身上身**（非仅单臂夹爪基准）。
  - **R2：** **IQL** 轨迹优化（Q 梯度修正动作后回灌 R1 数据再 SFT）+ **源噪声分布 actor** 在线 RL，避免直接对 flow 模型做 on-policy PG。

## 核心摘录（面向 wiki 编译）

### 1) 五阶段训练课程（L0→L1→R0→R1→R2）

- **链接：** <https://arxiv.org/html/2602.00919#S2>
- **摘录要点：**
  - **L0：** 现成大规模 VLM 底座（无机器人动作）。
  - **L1：** **24M** 网页多模态（RefSpatial、RoboPoint、ShareRobot、PixMo-Points、COCO、A-OKVQA、OpenSpaces 等）补 **物理/空间/affordance** 先验。
  - **R0：** **>3000 h** 跨人形/移动双臂/单臂数据统一预训练；**184M** 机器人域样本（含从 AgiBot 等构造的 VQA/轨迹预测）。
  - **R1：** 目标本体高质量 SFT（SDPA、减 denoising 步等效率调优）。
  - **R2：** **保守 RL 对齐**——BC 先验 + 奖励塑形，改善长程 ACL、恢复与精细接触。
- **对 wiki 的映射：**
  - [Green-VLA（论文实体）](../../wiki/entities/paper-greenvla-staged-vla-humanoid.md)
  - [VLA](../../wiki/methods/vla.md) — 分阶段「网页先验 → 机器人预训练 → 本体适配 → RL 后训练」的可复用配方

### 2) DataQA 与统一动作空间

- **链接：** <https://arxiv.org/html/2602.00919#S3>–<https://arxiv.org/html/2602.00919#S4.SS3>
- **摘录要点：**
  - 过滤缺失相机/帧、异常时长、低运动、高抖动、模糊帧、夹爪模式异常等。
  - **$\mathcal{A}_u$** 每维语义一致；**$m_e$** 掩码屏蔽未用槽位，消除 padding 虚假损失。
  - **速度条件调制** $v$：同模型覆盖精细接触与快速 reach/lift；推理时 $v$ 为可调超参。
  - **光流幅度** 估计各数据集执行速度并插值/降采样对齐。
- **对 wiki 的映射：**
  - [Green-VLA](../../wiki/entities/paper-greenvla-staged-vla-humanoid.md) — 与 [Foundation Policy Alignment](../../wiki/formalizations/foundation-policy-alignment.md) 的跨本体动作对齐实践对照

### 3) 推理栈：任务规划 + JPM 引导 + OOD 安全

- **链接：** <https://arxiv.org/html/2602.00919#S4.SS2>–<https://arxiv.org/html/2602.00919#S4.SS4>
- **摘录要点：**
  - **GigaVision** 将用户目标译为原子子任务序列（pick/place/give…），用 **episode-end** 与 VLM 反馈决定推进或重规划。
  - **JPM：** VLM 指 2D affordance → 深度反投影 3D → IK 得 $q^\star$ → **ΠGDM** 引导 flow matching 朝目标点。
  - **OOD：** 训练集状态 GMM 密度低于阈值时，沿 $\nabla p_{\text{train}}(s)$ 修正动作。
- **对 wiki 的映射：**
  - [Manipulation](../../wiki/tasks/manipulation.md) — 分层「慢规划 + 快 VLA 闭环」部署形态

### 4) R2：不直接 RL 梯度穿 flow 的两种对齐

- **链接：** <https://arxiv.org/html/2602.00919#S4.SS5>
- **摘录要点：**
  - **轨迹优化 + 原生微调：** IQL 学 Q；对 R1 rollout 动作加归一化 Q 梯度；环境验证后才并入数据集；从 R0 权重重启 R1 式微调。
  - **源分布优化：** 小 actor 学习 flow 初始噪声分布 $p_0'$，最大化环境回报，探索相对 BC 流形更保守。
- **对 wiki 的映射：**
  - [VLA](../../wiki/methods/vla.md) §部署经验后训练 — 与 ROVE / LWD 等「BC 饱和后对齐」并列的另一条 **flow-VLA** 路线

### 5) 实验要点（数字级）

- **链接：** <https://arxiv.org/html/2602.00919#S5>
- **摘录要点：**
  - **ALOHA 清理桌（R0，无额外微调）：** First-item SR **69.5%** vs π₀ **35.6%**；平均清理 **1m35s** vs π₀ **2m59s**（Table 2）。
  - **Simpler WidowX：** R2 相对 R1 **+24%** 绝对成功率；R2 平均 success **79.1%**（PaliGemma 3B 栈，Table 4）。
  - **Simpler Google Robot（Qwen3-VL R1）：** 平均 **71.8%**（Table 3）。
  - **CALVIN ABC→D：** R2 显著提升 **ACL** 与组合任务（Figure 14）。
  - **电商货架：** JPM 引导在 **SKU 级** 与 **OOD 包装** 上大幅提升 top-1（Figure 11）。
  - **Green 人形：** 双手 pick/place/handover/水果分拣/清理桌；OOD 场景布局仍保持任务跟随（Figure 12–13）。
- **对 wiki 的映射：**
  - [Green-VLA](../../wiki/entities/paper-greenvla-staged-vla-humanoid.md)

## 对 wiki 的映射

- 沉淀实体页：[Green-VLA 分阶段 VLA 与人形部署（arXiv:2602.00919）](../../wiki/entities/paper-greenvla-staged-vla-humanoid.md)
- 交叉补强：[VLA](../../wiki/methods/vla.md)、[Manipulation](../../wiki/tasks/manipulation.md)、[Behavior Cloning](../../wiki/methods/behavior-cloning.md)、[π₀.₇](../../wiki/methods/pi07-policy.md)

## 当前提炼状态

- [x] 五阶段课程、DataQA、统一动作、JPM/OOD、R2 双路线与主表结果摘录
- [x] wiki 实体页与 VLA 方法页交叉链接规划
- [x] 官方代码与项目页归档

## BibTeX

```bibtex
@misc{greenvla2026,
  title={Green-VLA: Staged Vision--Language--Action Model for Generalist Robots Manipulation},
  author={Sber Robotics Center},
  year={2026},
  eprint={2602.00919},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2602.00919},
}
```
