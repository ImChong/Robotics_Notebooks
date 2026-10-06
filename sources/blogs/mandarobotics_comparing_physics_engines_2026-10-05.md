# Comparing physics engines for robotics simulation — 来源归档

## 来源信息
- **发布者**：Manda Robotics
- **标题**：Comparing physics engines for robotics simulation
- **发布日期**：2026-10-05
- **原文**：https://mandarobotics.com/blog/comparing-physics-engines/index.html
- **配套背景文章**：https://mandarobotics.com/blog/comparing-physics-engines/primer.html
- **范围**：同一机器人、场景资产与控制器下，比较四种刚体/软体物理实现，评估接触、轨迹、重复性和吞吐；不与真机 ground truth 对照。

## 实验配置
| 实现 | 版本/路径 | 计算精度 |
|---|---|---|
| PhysX | Isaac Sim 6.1.0.0，CPU TGS | float32 |
| Newton + MuJoCo-Warp | Newton 1.5.2 + MuJoCo-Warp 3.11.0 + Warp 1.16.0，GPU | float32 |
| MuJoCo CPU | MuJoCo 3.11.0，native CPU | float64 |
| Genesis | Genesis 1.4.1，native rigid-body solver，GPU | float64 |

共同对象为 Franka Emika Panda。作者审计了导入模型的质量、质心、惯量、关节轴/限制、初始位姿与碰撞几何，并使用共同的显式力矩控制器；接触族并非完全相同：MuJoCo CPU、MuJoCo-Warp 和 Genesis 使用允许轻微穿透的软接触配置，PhysX 使用硬约束接触。软体路线还涉及不同的材料实现和刚柔耦合方式。

## 主要发现
- 低复杂度轨迹可以非常接近（如滑块停止端点约差 0.05 mm），但相似的整体运动会掩盖接触历史与力峰值差异；文章报告部分接触峰值最多相差约 10 倍。
- 多物体碰撞会积累为不同的物体路径；不拥挤或简单设置中的一致结果不能代表密集接触场景。
- 紧间隙 peg insertion、1 mm 布局扰动、不同控制器增益都会改变接触与跟踪表现；只看 task success 会漏掉失败严重度和厘米级轨迹差异。
- 测试软体抓取时，各实现出现不同的保持/下放/数值稳定结果；加密网格或缩小步长并未统一这些表现。PhysX 的软体初始化未通过，作者因此没有把它解读为 PhysX 软体能力结论。
- 未做真机对照，所以这些差异说明实现/设置不可互换，不能推出哪个引擎更接近现实。大多数场景是单次实验，重复试验仅用于部分灵敏度检查。

## 吞吐量证据边界
文章给出的单 Panda、无渲染、1 ms 控制步测试中，单环境 MuJoCo CPU 约 0.17M steps/s；MuJoCo-Warp 与 Genesis 单环境 GPU 约 0.006M 与 0.002M steps/s。批量测试中，8,192 个 world 时二者约 21.6M 与 11.8M steps/s，131,072 个 world 时约 35.2M 与 26.4M。文章注明 PhysX 未做批量测试，MuJoCo CPU 仅测试单 world（不是多核批量）；硬件、精度与执行路径不同，不能把数值当作各引擎的普遍排行榜。

## 可复核材料
原文每个场景链接到测量数据、audit/setup notes 和 replay；例子包括：
- 碰撞链 metrics：https://mandarobotics.com/blog/comparing-physics-engines/assets/collision-chain/metrics.json
- 碰撞链审计：https://mandarobotics.com/blog/comparing-physics-engines/assets/collision-chain/audit.md
- 推挤 metrics：https://mandarobotics.com/blog/comparing-physics-engines/assets/push/metrics.json
- runtime protocol：https://mandarobotics.com/blog/comparing-physics-engines/assets/runtime/RUNTIME-PROTOCOL-2026-09-25.md

## Wiki 映射
- 文章详情：[Manda Robotics 跨引擎物理仿真比较](../../wiki/entities/mandarobotics-physics-engine-comparison.md)
- 相关框架：[仿真器选型指南](../../wiki/queries/simulator-selection-guide.md)、[机器人仿真三层分工](../../wiki/concepts/robot-simulation-three-layers.md)
- 现有引擎详情：[MuJoCo](../../wiki/entities/mujoco.md)、[Isaac Sim](../../wiki/entities/isaac-sim.md)、[Newton Physics](../../wiki/entities/newton-physics.md)、[Genesis](../../wiki/entities/genesis-sim.md)
