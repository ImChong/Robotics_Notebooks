# PBHC（KungfuBot 官方实现）

- **标题：** PBHC — Physics-Based Humanoid Control
- **类型：** repo
- **仓库：** <https://github.com/TeleHuman/PBHC>
- **项目页（v1）：** <https://kungfubot.github.io/>
- **项目页（v2）：** <https://kungfubot2-humanoid.github.io/>
- **论文：** KungfuBot（NeurIPS 2025，arXiv:[2506.12851](https://arxiv.org/abs/2506.12851)）；KungfuBot 2 / VMS（ICRA 2026，arXiv:[2509.16638](https://arxiv.org/abs/2509.16638)）
- **机构：** 中国电信人工智能（TeleAI）、上海交通大学（SJTU）、华东理工大学（ECUST）、哈尔滨工业大学（HIT）、上海科技大学（ShanghaiTech）
- **收录日期：** 2026-07-09

## 一句话摘要

TeleHuman 开源的 **KungfuBot / KungfuBot2 统一代码库**：视频/LAFAN/AMASS → SMPL 统一格式 → Mink 或 PHC 重定向到 Unitree G1 → IsaacGym RL 模仿 → MuJoCo sim2sim / 真机部署；v1 强调高动态武术 **自适应跟踪课程 + 非对称 actor-critic**，v2（2025-10 起）在同一仓库支持 **VMS 通用多动作跟踪**（混合局部/全局目标、OMoE、段级奖励）。

## 为何值得保留

- **高动态 WBT 标杆栈：** NeurIPS 2025 KungfuBot 在 G1 真机复现踢腿、旋子、太极等高难动作；跟踪误差显著低于可部署基线，接近 oracle。
- **端到端可复现管线：** `motion_source/` → `smpl_retarget/` → `humanoidverse/` 模块化，含示例 motion 与预训练 checkpoint（`example/pretrained_hors_stance_pose/`）。
- **谱系延续：** 同一仓库 2025-10 扩展 **general motion tracking** 以支撑 KungfuBot2/VMS；与 ASAP、BeyondMimic、PHC、GVHMR、MaskedMimic 重定向栈显式对齐。
- **社区活跃：** GitHub 1000+ stars（2026-07）；CC BY-NC 4.0 许可。

## 仓库模块（编译自 README）

| 目录 | 作用 |
|------|------|
| `motion_source/` | 从视频、LAFAN、AMASS 等采集并统一到 SMPL |
| `smpl_retarget/` | 滤波、校正、重定向到 G1（**Mink** 或 **PHC** 管线可选） |
| `smpl_vis/`、`robot_motion_process/` | SMPL/机器人轨迹可视化、插值与分析 |
| `humanoidverse/` | IsaacGym 中 PPO（rsl_rl）训练；MuJoCo 部署与真机扩展接口 |
| `example/` | 实验样例 motion + 预训练策略 |

## 新动作接入流程

1. 采集源数据并处理为 SMPL（`motion_source/`）。
2. 重定向到机器人（`smpl_retarget/`，选 Mink 或 PHC）。
3. 可视化质检（`smpl_vis/`、`robot_motion_process/`）。
4. IsaacGym 训练策略（`humanoidverse/`）。
5. MuJoCo 或真机部署（`humanoidverse/`）。

## 对 Wiki 的映射

- [paper-notebook-kungfubot-physics-based-humanoid-whole-body-cont](../../wiki/entities/paper-notebook-kungfubot-physics-based-humanoid-whole-body-cont.md) — KungfuBot v1 论文+方法归纳
- [paper-notebook-kungfubot-2](../../wiki/entities/paper-notebook-kungfubot-2.md) — KungfuBot2 / VMS 论文+方法归纳
- [paper-kungfuathlete-humanoid-martial-arts-tracking](../../wiki/entities/paper-kungfuathlete-humanoid-martial-arts-tracking.md) — 同域武术高动态 tracking 姊妹对照
- [motion-retargeting](../../wiki/concepts/motion-retargeting.md) — Mink/PHC 重定向选型语境
- [humanoid-reference-motion-datasets](../../wiki/comparisons/humanoid-reference-motion-datasets.md) — LAFAN/AMASS 等输入数据源

## 参考来源（原始）

- 代码仓库：<https://github.com/TeleHuman/PBHC>
- KungfuBot 项目页：<https://kungfubot.github.io/>
- KungfuBot2 项目页：<https://kungfubot2-humanoid.github.io/>
