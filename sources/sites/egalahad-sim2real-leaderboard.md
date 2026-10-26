# sim2real Motion Tracking Leaderboard

> 来源归档

- **类型：** benchmark / project site
- **链接：** https://egalahad.github.io/sim2real/leaderboard/
- **访问日期：** 2026-10-26
- **一句话说明：** 在共享 MuJoCo 与冻结动作集上对多个公开人形跟踪策略进行同口径对比的交互排行榜。

## 榜单范围

- 页面列出 15 个策略发布/配置条目，归并为 10 个项目家族；下表在新实体中逐行映射到一个唯一的项目详情节点，版本条目共用同一项目页，不重复创建实体。
- 默认测试集是 MotionDecode Locomotion-80、Manipulation-48、Ground-60 和 Dance-40；Legacy LAFAN-40、PHUMA-30、Root-90 保留作历史对照。
- 协议采用 seed 0、50 Hz policy frame、initial pause 0、无 runtime cap 和共同失败条件。每个 dataset 内按 motion 等权，页面 headline 对可用的 dataset means 等权平均。
- 指标包含局部 body position/orientation、Locomotion 的 Global Root、GPU training hours、wrist position/orientation、tracking return 与 progress。位置/姿态误差越低越好，return/progress 越高越好。
- Global Root 只对 Locomotion 报告；Manipulation 指 body/wrist tracking，不代表物体操作任务成功率。GPU hours 仅在有可辩护来源时填入；该榜单不是对所有训练成本的完整统计。
- 页面标注最近更新时间为 2026-09-15；复现前应在官方页确认榜单数据及协议是否有更新。

## 榜单快照（页面标注：2026-09-15）

以下为页面当前排序前三行；首列为榜单展示值，括号内分别是 Locomotion 与 Manipulation 数据集均值。误差越低越好，Tracking Return 与 Progress 越高越好。GPU Hours 仅抄录页面已填且带来源链接的值。

| 排名 / 策略 | Body Pos (mm) | Body Ori (rad) | Wrist Pos (mm) | Tracking Return | Progress (%) | GPU Hours |
|---|---:|---:|---:|---:|---:|---:|
| 1 · Mimic Lite v1.1 | 26.68 (25.11 / 28.25) | 0.088 (0.081 / 0.095) | 22.86 (22.15 / 23.57) | 1.852 (1.913 / 1.790) | 96.5 (99.3 / 93.6) | 50.1 |
| 2 · MimicLite-ROA | 31.98 (29.22 / 34.74) | 0.106 (0.095 / 0.118) | 33.21 (31.61 / 34.81) | 1.826 (1.886 / 1.766) | 96.6 (98.7 / 94.5) | 173.2 |
| 3 · MimicLite-PPO | 32.92 (31.52 / 34.32) | 0.109 (0.097 / 0.121) | 35.96 (33.25 / 38.66) | 1.836 (1.900 / 1.772) | 97.6 (100.0 / 95.2) | 92.3 |

页面没有为这三行提供 Global Root 的 Manipulation 值；横评时保留为缺失值，不作补零。

## 复现

官方教程固定 MotionDecode revision、ordered motion list 与文件 hashes；默认评测 Locomotion-80 / Manipulation-48，可选择四组全评。结果应随代码 commit、依赖锁、策略配置与原始轨迹保存。旧版 LAFAN/PHUMA/Root90 不是 MotionDecode 的替代集合。
