# opendrivelab.com/RoboNaldo（RoboNaldo 项目页）

> 来源归档（ingest）

- **标题：** RoboNaldo — Humanoid Soccer Shooting
- **类型：** site / project-page
- **官方入口：** <https://opendrivelab.com/RoboNaldo/>
- **入库日期：** 2026-06-10
- **一句话说明：** 论文配套站点：强调 **通用场景下亚米级人形射门**、**单条人类参考 → 跟踪 → 偏离适应** 的三阶段课程，以及 **G1 机载感知 + 室外多场地** 真机演示（任意球 / 来球、多目标高度与球位 sweep）。

## 页面公开信息（检索自 2026-06-10）

| 资源 | URL |
|------|-----|
| 项目首页 | <https://opendrivelab.com/RoboNaldo/> |
| arXiv | <https://arxiv.org/abs/2606.11092> |

## 与论文一致的公开主张（便于 wiki 溯源）

1. **定位：** 「World's first less-than-1-meter-level accurate humanoid soccer shooting policy in general cases」——单人类参考，先 track 再 deviate/adapt。
2. **三阶段管线：** Stage 1 motion tracking → Stage 2 stationary-ball 射门适应 → Stage 3 启发式 kick-timing + locomotion 规划驱动 **one-touch** 来球射门。
3. **真机亮点：** 3 m 距离 **0.73 m / 0.86 m** 点级误差（静止 / 来球）；最佳 **17 cm @ 3 m**、**13.10 m/s** 球速；**egocentric onboard** 球与目标感知。
4. **演示矩阵：** 低/中高/超高目标 × 左中右；球位横向与远近 sweep；来球速度 sweep；人工草 / 曲棍球场 / 天然草等室外场地。
5. **仿真：** 3 m 射门距离、8 m×2 m 目标面 **shot-quality heatmap**；Stage 2 覆盖任意球 regime，Stage 3 略牺牲精度换来球泛化。

## 对 wiki 的映射

- [`wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md`](../../wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md) — 方法栈、实验、与 PAiD 等对照及部署归纳
