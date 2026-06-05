# Explicit Stair Geometry Conditioning for Robust Humanoid Locomotion（arXiv:2605.09944）

> 论文来源归档（ingest）

- **标题：** Explicit Stair Geometry Conditioning for Robust Humanoid Locomotion
- **类型：** paper / humanoid locomotion / perceptive RL / explicit terrain representation
- **arXiv：** <https://arxiv.org/abs/2605.09944>（HTML：<https://arxiv.org/html/2605.09944v1>）
- **PDF：** <https://arxiv.org/pdf/2605.09944.pdf>
- **机构：** 深圳市人工智能与机器人研究院（AIRS）、香港中文大学（深圳）、Mohamed bin Zayed University of Artificial Intelligence（MBZUAI）
- **平台：** Unitree G1（实机）；Isaac Lab（训练）、MuJoCo（跨仿真评测，无微调）
- **入库日期：** 2026-06-05
- **一句话说明：** 用 **BEV 点云编码器** 预测 **可解释的楼梯几何 token**（地形类别 + 踢面高度 + 踏面深度 + 相对航向），直接 **条件化 PPO 全身策略**，在仿真中相对盲走 / 高程图 / 视觉 MoRE 基线取得更高成功率与 **训练分布外踢面高度泛化**；**Unitree G1** 室内外楼梯零样本部署，户外 **连续 33 级** 上楼无失败。

## 核心论文摘录

### 1) 问题：隐式地形表征 vs 盲本体策略

- **要点：** 楼梯的几何不连续与踢面高度敏感使 **可靠爬梯** 难于平地速度跟踪；**盲走** 仅靠接触反馈 **缺乏前瞻**，摆腿净空与步幅调制常不足；**感知方法** 多把地形压成 **高维隐式 latent**（高程图、深度图），对噪声敏感、难解释，且 **超出训练分布的楼梯几何** 时性能骤降。
- **对 wiki 的映射：** `wiki/entities/paper-explicit-stair-geometry-humanoid-locomotion.md`；交叉补强 `wiki/concepts/terrain-adaptation.md`

### 2) 显式楼梯几何 token：BEV 感知 → 四维结构化表征

- **要点：** 机载 **机器人坐标系点云** 投影为 **3 m×3 m、5 cm 分辨率** 的 **6 通道 BEV**（每格 max/min/mean z、高差、std、点密度）；CNN 编码至 **128×8×8** 特征后回归 **楼梯状态** $s_t\in\{\text{flat},\text{stairs-up},\text{stairs-down}\}$ 与 **$h_{\text{step}}, d_{\text{step}}, \theta_{\text{yaw}}^{\text{current}}$**，组成 **$z_t\in\mathbb{R}^4$** 直接拼入策略观测 $o_t=\{o_t^{\text{prop}}, z_t\}$；平地时 $h=d=0$。
- **对 wiki 的映射：** 同上实体页；`wiki/tasks/locomotion.md`、`wiki/methods/reinforcement-learning.md`

### 3) 三阶段训练：特权教师 → 感知学生 → 联合 PPO

- **要点：** **阶段 1** 用仿真 **真值几何参数** 预训练 locomotion；**阶段 2** 在教师监督下独立训感知网络（分类 CE + 高度/深度 SmoothL1，$\lambda_{\text{cls}}=0.6,\lambda_h=\lambda_d=1$）；**阶段 3** **$\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{PPO}}+\alpha\mathcal{L}_{\text{terrain}}$**（$\alpha=1$）联合优化；部署 **仅保留学生编码器 + 策略**，无特权信息。奖励沿用 **Isaac Lab rough-locomotion** 默认加权项；动作为 **全身 PD 目标关节**。
- **对 wiki 的映射：** 同上实体页；`wiki/concepts/privileged-training.md`、`wiki/entities/isaac-gym-isaac-lab.md`

### 4) 实验：仿真消融、OOD 踢面高度、G1 长程户外

- **要点：** 统一 PPO 下对比 **Blind-PPO**、**HeightMap-PPO（11×17 高程图）**、**本文**；Table I：成功率 **52% → 88% → 96%**（3 seed）。几何估计：仿真 MAE **0.6/0.7 cm**（$h/d$）、真机 **0.9/1.1 cm**，状态分类 **99.1% / 97.3%**。踢面 **0.12–0.16 m 训练、0.18–0.22 m 测试**：相对 **MoRE** 视觉基线，0.22 m 成功率 **82.7% vs 69.4%**（+13.3 pp）。不规则户外楼梯成功率 **93%**（Blind 20%、HeightMap 75%、MoRE 72%）。实机：**室内 5 级** 连续上楼；**户外 33 级** 连续上楼；MuJoCo **变速跟踪** 无微调评测。
- **对 wiki 的映射：** 同上实体页；对照 `wiki/entities/paper-faststair-humanoid-stair-ascent.md`、`wiki/entities/unitree-g1.md`、`wiki/concepts/sim2real.md`

## 相关资料

- **同主题楼梯学习：** [FastStair（arXiv:2601.10365）](faststair_arxiv_2601_10365.md) — DCM 落脚点规划 + LoRA 专家融合（LimX Oli）
- **同平台感知行走：** [E-SDS（arXiv:2512.16446）](e_sds_arxiv_2512_16446.md) — VLM 环境统计奖励 + 高度图/LiDAR（G1）
- **MoRE 基线（论文引用 [28]）：** 复杂地形多专家 lifelike 步态；知识库见 `wiki/entities/paper-amp-survey-08-more.md`
- **公开代码 / 项目页：** 截至 ingest 日，arXiv 条目未附官方 GitHub 或项目主页链接

## 当前提炼状态

- [x] 摘要与主方法摘录
- [x] wiki 页面映射确认
