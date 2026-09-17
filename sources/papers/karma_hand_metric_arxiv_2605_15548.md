# A Kinematic Metric for Fine Manipulation Ability in Robotic Hands（KaRMA）

> 来源归档

- **标题：** A Kinematic Metric for Fine Manipulation Ability in Robotic Hands
- **类型：** paper
- **作者：** Martin Peticco, Pulkit Agrawal
- **机构：** Improbable AI Lab, Massachusetts Institute of Technology（MIT）
- **会议：** IEEE/RSJ IROS 2026
- **arXiv：** [2605.15548](https://arxiv.org/abs/2605.15548)
- **项目页：** <https://martinpeticco.com/karma/>
- **代码：** <https://github.com/mfpeticco/karma-hand-metric>
- **入库日期：** 2026-09-17
- **一句话说明：** 仅 URDF 运动学、无控制器/策略，在拇指–食指 rolling pinch 上 BFS 平移 + HEALPix 228 姿态分区旋转探索，输出 KaRMA-T/R/S 三分数；16 手 leaderboard 显示平移与旋转可解耦、DoF/Jacobian 代理指标易误判。

## 开源核查（2026-09-17）

| 项 | 结论 |
|----|------|
| GitHub `mfpeticco/karma-hand-metric` | **已开源**：`run_metric.py`、`run_all_hands.py`、16 手配置、论文结果 `results/16_hand_batch/` |
| 依赖 | Python 3.10 conda；`pin`（Pinocchio）、hpp-fcl；**CPU-only** |
| 数据 |  bundled 16 手 URDF + 预计算 pkl/yaml |

## 核心摘录（对用户三阶段 + 官方第四阶段）

1. **Seed grasps** — 确定性生成候选 thumb–index pinch，IK 投影到双接触流形；每个可行 seed 均参与。
2. **Translation search** — 每 seed BFS 沿抓取主轴滚动球体；每步解 rolling-contact QP，检 joint limits / collision / antipodal force。
3. **Rotation exploration** — 每到达位姿绕两可控轴 tilt，统计 **228** 个 HEALPix 等面积姿态 bin 可达比例。
4. **Scoring** — 可达体积 → **KaRMA-T**；orientation coverage → **KaRMA-R**；跨 seed  spread → **KaRMA-S**；长度按 hand-size 常数无量纲化。

## 16 手 Table I 要点（论文复现值）

| 手 | DOF | KaRMA-T | KaRMA-R | KaRMA-S |
|----|-----|---------|---------|---------|
| LEAP | — | **0.097**（最高 T） | 次高 R 梯队 | — |
| Allegro | — | 第二 T | **0.335**（最高 R） | median≈>0.5 best |
| D'Claw | 6 | 0.036 | 0.231 | 低 S（依赖窄窗口 seed） |
| Shadow | 9 | 0.013 | — | — |
| xHand1 | 6 | 0.004 | — | 91% voxel 单层 |
| Ability | — | 末位 | GCI 高但 KaRMA 末位 | — |

- T 与 R 总体 Spearman **ρ=0.96**，但中段手任务相关分化（Allegro 善 R、LEAP 善 T）。
- Workspace opposability vs KaRMA-T **ρ=0.93**；Yoshikawa **ρ=−0.20**；GCI **ρ=−0.08**。
- 约束消融：joint limits  alone 可削 58–99% naive workspace（LEAP 0.580→0.241）。

## 对 wiki 的映射

- 实体页：[KaRMA](../../wiki/entities/paper-karma-hand-metric.md)
- 交叉：[HAND ERC 灵巧评测](../../wiki/entities/paper-hand-erc-benchmarking-dexterity.md)、[Allegro Hand](../../wiki/entities/allegro-hand.md)、[DexBench](../../wiki/entities/dexbench.md)、[in-hand reorientation](../../wiki/methods/in-hand-reorientation.md)
