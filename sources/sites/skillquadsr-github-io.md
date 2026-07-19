# skillquadsr.github.io（APT-RL / Science Robotics 2026 封面）

- **标题：** Agile perceptive multi-skill locomotion for quadrupedal robots in the wild — 官方项目页
- **类型：** site / project-page
- **URL：** <https://skillquadsr.github.io/>
- **入库日期：** 2026-07-19
- **配套论文：** [Science Robotics 2026 封面](https://doi.org/10.1126/scirobotics.adz7397) · [arXiv:2607.13579](https://arxiv.org/abs/2607.13579) — 归档见 [`sources/papers/apt_rl_science_robotics_2026.md`](../papers/apt_rl_science_robotics_2026.md)

## 一句话摘要

KAIST 等团队提出的 **APT-RL**（Action Pretrained Transformer-based Reinforcement Learning）官方站点：展示 **轨迹优化预训练 + TVAE 力矩先验 + PPO 辅助动作 + 深度/LiDAR 感知蒸馏** 的统一四足控制栈，以及在 **KAIST HOUND** 上校园/森林 **1.1 km / 0.34 km** 户外零样本实机与 **6 m/s** 瞬时峰值速度演示。

## 公开信息要点（截至入库日）

- **机构：** 韩国国防科学研究所（ADD）、KAIST 机械工程系、DIDEN Robotics、高丽大学机械工程 / 智能移动学院（* Jun-Gill Kang & Jaehyun Park 同等一作；‡ Seungwoo Hong & Hae-Won Park 通讯作者）。
- **方法三阶段（主页）：**
  1. **Representation learning** — 基于 **2D 轨迹优化** 生成 **18 万条** 轨迹（约 15.5 h），用 **Transformer VAE（TVAE）** 学习潜空间与 **trot/bound 力矩解码器**。
  2. **Reinforcement learning** — 策略输出 **潜动作 + 辅助动作**，在 Isaac Gym 复杂地形上 PPO 训练；**Auto** 控制器相对固定 trot/bound 在成功率与能效上更优。
  3. **Perceptual distillation** — 特权高程 teacher → **RealSense D435 深度 + 2D LiDAR + 本体** 学生编码器（CNN-GRU）；**>4 m/s** 时在 LiDAR 与机体间加装 **3D 打印减振器** 抑制 **>10 g** 冲击。
- **实机平台：** **KAIST HOUND** 自研四足；**仅机载感知与算力**，无动捕 / 外部位姿估计。
- **野外路线：** 校园 **1.1 km**（多级楼梯、草地、坡道）；森林 **0.34 km**（倒木、树根、湿滑落叶等）；障碍含 **60 cm 高台、三级楼梯、栏、垫脚石、沟、倒枝** 等。
- **技能与步态：** 单策略内 **trot ↔ bound** 及 **走/跑/跳/攀台/落地** 等机动；同楼梯上 **1.8 m/s trot** vs **4.3 m/s bound**；几何相似下降障碍按 **高度** 选 gait。
- **页面链接：** Paper（Science）、arXiv HTML、Summary Video；**未列出 GitHub / Hugging Face 训练代码仓库**。
- **数据开放（论文 Data availability）：** 图表生成与数据已 deposit 至 [Zenodo:20645964](https://zenodo.org/records/20645964)（**非** 完整训练/部署代码栈）。

## 源码开放核查（步骤 2.5）

| 类别 | 状态 | 说明 |
|------|------|------|
| 训练/推理/部署代码 | **未开源** | 项目页头部仅有 Paper / arXiv / Video，**截至 2026-07-19 无 GitHub 链接** |
| 数据与作图代码 | **部分开放** | 论文写明 Zenodo [20645964](https://zenodo.org/records/20645964) 含复现结论所需数据与 figure generation code |
| 权重 / 数据集 | **部分开放** | 同上 Zenodo；不含完整 RL 训练栈 |

## 为何值得保留

- **非 PDF 证据：** 项目页视频直观呈现 **野外长程、步态切换、减振 LiDAR** 与 **同地形不同速度选 gait** 等论文难静态传达的细节。
- **与 arXiv / DOI 三角互证：** 方法框图、三阶段管线、HOUND 传感配置与野外里程数与 [`apt_rl_science_robotics_2026.md`](../papers/apt_rl_science_robotics_2026.md) 一致。
- **四足多技能感知 locomotion 谱系锚点：** 在 [Walk These Ways](../../wiki/entities/paper-walk-these-ways-quadruped-mob.md)（MoB 参数化）与 [Learning to Adapt](../../wiki/entities/paper-learning-to-adapt-bio-inspired-quadruped-gait.md)（盲本体 8 步态）之间，补齐 **深度+LiDAR 感知 + 高速 bound + 野外长程** 的 Science Robotics 级实证。

## 关联资料

- 论文归档：[`sources/papers/apt_rl_science_robotics_2026.md`](../papers/apt_rl_science_robotics_2026.md)
- Wiki 实体：[`wiki/entities/paper-apt-rl-agile-perceptive-quadruped-locomotion.md`](../../wiki/entities/paper-apt-rl-agile-perceptive-quadruped-locomotion.md)
