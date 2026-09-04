# World-Model-Augmented Visual Locomotion for Humanoids on Foothold-Constrained Terrain（arXiv:2609.02542）

> 来源归档（ingest）

- **标题：** World-Model-Augmented Visual Locomotion for Humanoids on Foothold-Constrained Terrain
- **简称：** WM-LOCO
- **类型：** paper / humanoid / world-model / perceptive-locomotion / foothold
- **arXiv：** <https://arxiv.org/abs/2609.02542>
- **PDF：** <https://arxiv.org/pdf/2609.02542>
- **HTML：** <https://arxiv.org/html/2609.02542>
- **项目页：** <https://m0puppet.github.io/wm-loco/> — 归档见 [`sources/sites/wm-loco.md`](../sites/wm-loco.md)
- **代码：** 截至 2026-09-04 **待发布**（项目页 *Code (coming soon)*；`M0PUPPET/wm-loco` 仅静态页）
- **机构：** 地瓜机器人（D-Robotics）；北京邮电大学（BUPT）；苏州大学（Soochow）；哈尔滨工业大学（HIT，论文作者页列出 Ao Zhang 第二机构）
- **入库日期：** 2026-09-04
- **最后更新：** 2026-09-04
- **一句话说明：** 在稀疏落脚地形上，把 RSSM 世界模型与 PPO 联合训练，用预测性循环特征引导单深度人形策略；G1 机载 93.3% 平均成功。

## 开源状态（步骤 2.5，2026-09-04）

| 组件 | 状态 |
|------|------|
| 项目页 | 已上线 |
| 训练 / ONNX / 权重 | **未见**（页上 Code coming soon） |

**结论：待发布**

## 核心论文摘录（MVP）

### 1) 预测性循环特征，而不是显式落脚标签

- 观测：5 帧本体历史 + 单头戴深度 + 基座速度命令；RSSM（确定性记忆 \(h_t\) + 128 维随机潜变量 \(z_t\)）重建本体/深度/奖励，并把 \(f_t^{\mathrm{WM}}=g_\psi(h_t)\) 交给 MoE actor–critic。
- 与 PPO 基线共享奖励、深度、AMP 先验、MoE 与迭代预算；基线只去掉世界模型通路。
- **对 wiki 的映射：** [paper-wm-loco](../../wiki/entities/paper-wm-loco.md)、[Humanoid Locomotion](../../wiki/tasks/humanoid-locomotion.md)、[楼梯与障碍感知移动](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)

### 2) 稀疏落脚上 PPO 基线归零，世界模型拉开差距

- IsaacLab 8192 env、RTX 5880 Ada；楼梯/沟/踏石 Easy–Hard：PPO 在沟与踏石 **0%**，WM-LOCO 沟 90–100%、踏石 78.2–88.3%、楼梯 92.0–95.7%。
- 楼梯步态：步幅 +15–35%、骨盆加速度 −24–33%、机械能 −6–20%。
- **对 wiki 的映射：** [paper-wm-loco](../../wiki/entities/paper-wm-loco.md)、[P³](../../wiki/entities/paper-p3.md)、[Hiking in the Wild](../../wiki/entities/paper-hiking-in-the-wild.md)

### 3) 同一策略 ONNX 上 G1，无离板感知、无地形图

- Jetson Orin + RealSense D435；踏石 100% / 楼梯 90% / 0.8 m 沟 90%（各 10 trial，平均 **93.3%**）。
- **对 wiki 的映射：** [paper-wm-loco](../../wiki/entities/paper-wm-loco.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)

## 当前提炼状态

- [x] 论文摘要与表格摘录
- [x] 项目页核查（步骤 2.5）
- [x] wiki 映射
