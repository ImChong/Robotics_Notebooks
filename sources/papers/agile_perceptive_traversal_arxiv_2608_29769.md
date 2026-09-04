# Agile Perceptive Traversal / Sparse 3D Structures（arXiv:2608.29769）

> 来源归档（ingest）

- **标题：** Learning Agile Perceptive Traversal of Sparse 3D Structures for Humanoids
- **类型：** paper / humanoid / perceptive-locomotion / brachiation / lidar / teacher-student
- **arXiv abs：** <https://arxiv.org/abs/2608.29769>
- **PDF：** <https://arxiv.org/pdf/2608.29769>
- **HTML：** <https://arxiv.org/html/2608.29769>
- **项目页：** <https://nemantor.github.io/sparse-3d-traversal-website/>
- **机构：** 苏黎世联邦理工（ETH Zürich）Robotic Systems Lab；ETH AI Center；Computer Vision and Geometry Group
- **作者：** Efe Ongan、Chong Zhang、Boyang Sun、Andrei Cramariuc、Cesar Cadena、Marco Hutter
- **发表 / 上传：** 2026-08（arXiv）
- **平台：** ENGINEAI PM-01 + RoboSense E1R 固态 LiDAR + 被动钩式末端
- **入库日期：** 2026-09-02
- **再核日期：** 2026-09-04

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| arXiv | [2608.29769](https://arxiv.org/abs/2608.29769) | 论文与附录 |
| 项目页 | [sparse-3d-traversal-website](https://nemantor.github.io/sparse-3d-traversal-website/) | 真机视频、方法交互图、注意力可视化 |
| 前作 | [AME-2 arXiv:2601.08485](https://arxiv.org/abs/2601.08485) | 注意力地图编码器直接用于原始 LiDAR 栅格 |
| 对照 | [LadderMan arXiv:2606.05873](https://arxiv.org/abs/2606.05873) | 人形感知梯子攀爬（深度而非原始 LiDAR） |

## 开源状态（步骤 2.5，2026-09-04 再核）

- **确认未开源：** 项目页 **无** GitHub / Hugging Face / Zenodo 链接；页脚无 Code 区。作者 Pages 账号 [`nemantor`](https://github.com/nemantor) 仅有无关 Java/LIVO fork，**无** 本项目训练/部署仓。
- **处理：** wiki 标未开源；`## 源码运行时序图` 标不适用。勿建 `sources/repos/`。

## 摘要级要点

- **问题：** 稀疏薄型 3D 结构（猴架横杆 1–3 cm 半径）需 onboard 感知 + 爆发式全身接触序列；高程图丢悬空结构，体素代价高，现有 raw-depth/lidar 工作多面向稠密地形。
- **方法：** 分阶段特权教师（跳上/荡杆/跳下）+ **AME-2 注意力编码** 原始 E1R 点云 + **GRU** 时序融合 + 辅助梯中心线预测；三阶段蒸馏（DAgger → critic warm-up → 正则 PPO）。
- **Sim2Real：** 电池压降、执行器热限、E1R 射线发散/边缘 dropout/虚假回波建模；被动钩 + 腕 yaw 脱钩。
- **真机：** 完整 jump-up→brachiation→jump-down **14/15**（93%）；荡杆 **0.5 m/s**；矮身 2×2 cm 杆 **10/10**。
- **消融：** AME-2（13.8k 参数）蒸馏 BC loss 最低；盲学生最差，确认必须外感知。

## 核心摘录（面向 wiki 编译）

### 1) 感知学生（§IV-F）

- 观测：本体 4 帧历史 + 平面目标命令 + E1R 36×35×4（10 Hz）；AME-2 在 2D 栅格上做注意力，非无序 PointNet。
- GRU 融合感知特征与本体嵌入；辅助头预测最近横杆相对位姿（训练期）。

### 2) 真机结果（Table V 节选）

| 配置 | 高度 h [m] | 间距 s [m] | 完整序列 |
|------|-----------|-----------|---------|
| 梯 A (9) | 1.69 | 0.26 | 100% |
| 梯 B (2) | 1.72 | 0.31 | 100% |
| 梯 C (4) | 1.75 | 0.33 | 75% |
| 合计 | — | — | **14/15** |

### 3) 编码器消融（Table IV，BC loss ×10⁻²）

| 编码器 | 参数量 | total | jump | brach. | down |
|--------|--------|-------|------|--------|------|
| AME-2 + aux | 13.8k | **2.35** | **2.48** | **1.95** | **3.20** |
| AME-2 | 13.8k | 2.43 | 2.56 | 2.00 | 3.26 |
| CNN | 106.7k | 2.62 | 2.73 | 2.16 | 3.57 |
| MLP | 1.31M | 2.76 | 2.92 | 2.28 | 3.62 |
| Blind | — | 2.90 | 3.18 | 2.30 | 3.51 |

### 4) 电池 / 热 / LiDAR 噪声（§V）

- 跳上峰值机械功率 **2.11 kW**，电压最低 **34.7 V**；无压降模型会 brownout。
- 电池：\(V_{\mathrm{nom}}=51\,\mathrm{V}\)，\(V_{\mathrm{min}}=30\,\mathrm{V}\)，\(k_{\mathrm{sag}}=0.0375\)，\(T_{\mathrm{rec}}=0.1\,\mathrm{s}\)。
- 热积分：18 个低力矩关节，\(T_{\mathrm{ch}}=1\,\mathrm{s}\)，\(T_{\mathrm{leak}}=5\,\mathrm{s}\)，持续 \(\rho\ge 0.2\) 饱和。
- E1R 训练噪声：\(0.625^\circ\) 锥 + \(\sigma=2\,\mathrm{cm}\)；边缘 drop 0.05 / mix 0.20；距离门 \([0.3,1.5]\,\mathrm{m}\)。MuJoCo 验证：每像素 16 射线逆平方融合。

## 对 wiki 的映射

- 沉淀实体页：[Agile Perceptive Traversal](../../wiki/entities/paper-agile-perceptive-traversal-sparse-3d.md)
- 交叉补强：[AME-2](../../wiki/entities/paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi.md)、[LadderMan](../../wiki/entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md)、[PHP](../../wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)、[ANYmal Parkour](../../wiki/entities/paper-notebook-anymal-parkour-robust-perceptive-locomotion.md)、[stair-obstacle perceptive locomotion](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)

## 当前提炼状态

- [x] arXiv HTML 方法 / Table I–V / 硬件与 §V 噪声摘录
- [x] 项目页开源核查（2026-09-02 初核、2026-09-04 再核）：无代码链接
- [x] 升格 `wiki/entities/paper-agile-perceptive-traversal-sparse-3d.md`
- [x] 2026-09-04：枢纽页挂接 + 电池/热/射线锥参数编译入 wiki
