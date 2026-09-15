# legbot-MPC-WBC（四足 Convex MPC + WBC 参考实现）

- **标题：** legbot-MPC-WBC / go2-convex-mpc
- **类型：** repo
- **仓库：** <https://github.com/Robot-Nav/legbot-MPC-WBC>
- **许可：** MIT
- **硬件：** 最初 Unitree Go2，后适配 **LegBot** 四足；**sim2sim（MuJoCo）+ sim2real**
- **收录日期：** 2026-09-15
- **开源结论：** **已开源**（`main`：Convex MPC；`WBC` 分支：改进 MPC-WBC 框架）

## 一句话摘要

基于 **MIT Cheetah 3 Convex MPC** 的四足控制栈：**Pinocchio** 做运动学/动力学/质心/足端雅可比，**CasADi + OSQP** 解接触力 QP，**MuJoCo** 仿真；`main` 为凸 MPC 基线，`WBC` 分支叠加改进 **MPC–WBC** 分层。

## 为何值得保留

- **MPC→WBC 工程参考：** 与 [mpc-wbc-integration](../../wiki/concepts/mpc-wbc-integration.md)、[srbd-convex-mpc-wbc](../../wiki/concepts/srbd-convex-mpc-wbc.md) 知识链对齐；虽为 **四足** 而非人形，但 **MPC 规划接触力 + 低层执行** 的分层读法可迁移到人形 locomotion 教学。
- **可复现 sim2real：** README 记录实机控制环 ~15 Hz→优化后 30–40 Hz 的工程权衡。
- **算法出处明确：** 引用 Kim et al. MIT Cheetah 3 convex MPC 论文。

## 技术要点（编译自 README）

| 项 | 内容 |
|----|------|
| MPC | 质心动力学线性化 → 凸 QP 优化地面接触力；滚动执行首步 |
| 摆腿 | Raibert 落点 + 五次摆腿轨迹 |
| 仿真 | MuJoCo 1 kHz 物理 / 200 Hz 腿控 / 30–50 Hz MPC |
| 能力 | Trot ~3 Hz；前进最高 ~0.8 m/s（Go2 仿真） |
| 分支 | `main`：Convex MPC；`WBC`：改进 MPC-WBC |

## 对 Wiki 的映射

- [mpc-wbc-integration](../../wiki/concepts/mpc-wbc-integration.md)
- [legbot-mpc-wbc 实体页](../../wiki/entities/legbot-mpc-wbc.md)
- 人形 WBC 经典线：[hub-wbc](../../wiki/overview/hub-wbc.md)（注明四足参考实现）

## 参考来源（原始）

- 代码：<https://github.com/Robot-Nav/legbot-MPC-WBC>
- 算法参考：MIT Cheetah 3 Convex MPC（Kim et al.）
