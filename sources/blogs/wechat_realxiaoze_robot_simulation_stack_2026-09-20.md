# wechat_realxiaoze_robot_simulation_stack_2026-09-20

> 来源归档（blog / 微信公众号）

- **标题：** Isaac Sim/Lab、MuJoCo、Genesis、mjlab：机器人仿真走到哪一步了？
- **类型：** blog
- **作者：** RealXiaoze（具身智能研究室；Humanoid Motion Intelligence 知识库）
- **原始链接：** https://mp.weixin.qq.com/s/P9O1o6XME9wlTHlGGNLj4Q
- **入库日期：** 2026-09-20
- **抓取方式：** WebFetch（`mp.weixin.qq.com`；本环境未预装 `wechat-article-for-ai`）
- **原始抓取落盘：** [`sources/raw/wechat_realxiaoze_robot_simulation_stack_2026-09-20.md`](../raw/wechat_realxiaoze_robot_simulation_stack_2026-09-20.md)
- **关联 GitHub：** <https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main>
- **一句话说明：** 用「物理引擎 / 场景平台 / 学习框架」三层划分 Isaac、MuJoCo、Genesis 生态；区分吞吐量与单步延迟、软接触与软体建模、Sim2Real 三问验收与 data-driven 仿真趋势；给出按任务选工具的四问清单。
- **步骤 2.5（开源核查）：** 文内推广 [RealXiaoze/humanoid-motion-intelligence](https://github.com/RealXiaoze/humanoid-motion-intelligence)（策展知识库，非仿真引擎）；提及 Unitree Isaac Lab/mjlab 部署项目、Genesis-Humanoid、MuJoCo Playground 真机实验均为公开仓库/论文，无单一厂商项目页需核查。
- **沉淀到 wiki：** [`wiki/concepts/robot-simulation-three-layers.md`](../../wiki/concepts/robot-simulation-three-layers.md)

## 核心摘录（归纳，非全文）

### 三层分工（选型起点）

| 层次 | 处理什么 | 代表工具 |
|------|----------|----------|
| **物理计算** | 动力学、碰撞、接触、材料形变 | MuJoCo、PhysX、Newton、Genesis 内部求解器 |
| **场景与系统仿真** | 机器人/物体、渲染、传感器、控制连接 | Isaac Sim、Gazebo、SAPIEN、Genesis |
| **学习与任务组织** | 观测、动作、奖励、随机化、训练评测 | Isaac Lab、mjlab、MuJoCo Playground、ManiSkill |

同一产品可跨层；PhysX 等底层引擎也可被多平台复用。

### Isaac Sim vs Isaac Lab

- **Isaac Sim：** 场景构建、物理、渲染、传感器、控制联调；可只做系统验证、不必训练 policy。
- **Isaac Lab（2.x 主流）：** 在 Sim 之上组织 RL/IL 工作流（观测/奖励/重置）；回答「机器人在世界里怎样学」。
- **Isaac Lab 3.0 EA：** 拆分物理/渲染/可视化，支持 PhysX、Newton/MuJoCo-Warp；部分流程可不启完整 Sim——**早期访问，任务兼容性须自验**。

### 并行、框架与「快」的定义

- 训练 policy 需要高 **采样量**；MJX、MuJoCo Warp、Isaac 并行环境服务此需求。
- **总吞吐量**（多 env 合计 steps/s）≠ **单 env 单步延迟**；完整训练还含渲染、推理、参数更新、重置。
- **更有意义的指标：** 同等任务要求下，多久得到 **通过真机测试** 的 policy。
- **学习框架价值：** Isaac Lab、mjlab 等把 obs/reward/curriculum 模块化，避免每任务复制一套代码。

### 操作学习与软体

- **视觉操作：** 对象/场景多样性、渲染、数据采集（ManiSkill、Isaac 遥操作与示范生成）。
- **软接触 vs 软体：** 软接触 = 接触力–压入关系，物体仍可刚体；软体 = 物体自身形变（海绵、线缆、布料）。
- **MuJoCo 3.0 flex / cable；Genesis FEM/MPM/粒子。** GPU 后端 parity 须查：MJX 仍不支持 flex，Warp flex 完善中。
- 软体任务要把 **材料测量**（弯曲刚度、摩擦等）纳入研发，DR 只调刚体参数补不上形变缺失。

### Sim2Real 验收三问

1. **物理过程能否对得上**（滑移、夹稳、接触）
2. **真机能否完成任务**（成功率、误差阈值）
3. **换条件后能否稳定**（换物体/地面、连续运行、失败恢复）

**Playground 真机样例（须连同条件读）：**

| 任务 | 结果 | 条件 |
|------|------|------|
| Franka 物块姿态调整 | 35 次中 85.7% | 位置 3 cm、角度 10° 阈值 |
| Franka 图像抓方块 | 12/12 | 2D 平面、简化动作空间 |
| LEAP 手转方块 | 中位 3.5 次旋转后失败 | 连续操作仍易卡住 |

**零样本** 通常指真机上无继续学习；团队仍可能做过硬件测量与模型校准。

### Data-driven 仿真与通用模型

- Sergey Levine 担忧：更强模型可能更充分学到 **仿真偏差**；胡渊鸣倾向 **data-driven 仿真**（材料/接触从真机数据学）。
- **PhysTwin：** 少视角 RGB-D 视频 → 可变形对象 + 弹簧–质量参数优化。
- **GPT-6 Astra + Robocurve：** 通用模型可接真机动作接口；精密操作瓶颈不能由单次成功率定位。

### 选型四问 + 共享资产

1. 第一个任务多久跑通？
2. 失败能否定位？
3. 换设备要改什么？
4. 模型与部署代码是否有人维护？

**排错时间计入平台成本。** 值得共享：校准模型、可复现 train/deploy 配置、真机失败固定测试用例。

## 对 wiki 的映射

- [robot-simulation-three-layers](../../wiki/concepts/robot-simulation-three-layers.md)（本次升格主页面）
- [robot-training-stack-layers-technology-map](../../wiki/overview/robot-training-stack-layers-technology-map.md)（六层训练栈；与本页三层正交互补）
- [simulator-selection-guide](../../wiki/queries/simulator-selection-guide.md)（locomotion 三选一 + 分层入口）
- [simulation-evaluation-infrastructure](../../wiki/concepts/simulation-evaluation-infrastructure.md)（评测基础设施）
- [sim2real](../../wiki/concepts/sim2real.md)（迁移主线）
- [humanoid-motion-intelligence](../../wiki/entities/humanoid-motion-intelligence.md)（同源 GitHub 知识库）

## 可信度与使用边界

- 第三方工程归纳（RealXiaoze / 具身智能研究室），工具版本与 benchmark 以各项目官方为准。
- Isaac Lab 3.0、MJX flex parity、Playground 真机数字会随版本变化；引用时注明条件与日期。

## 当前提炼状态

- [x] 文章基础摘要填写
- [x] 初步 wiki 页面映射确认
