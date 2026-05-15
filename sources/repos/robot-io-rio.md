# RIO（Robot I/O）

> 来源归档

- **标题：** RIO: Flexible Real-Time Robot I/O for Cross-Embodiment Robot Learning
- **类型：** repo + paper（arXiv）+ 官方站点 + 文档站
- **组织：** CMU / Bosch Center for AI / TU Delft / Lavoro AI 等（论文作者单位）
- **代码：** https://github.com/robot-i-o/rio
- **硬件与底层驱动仓（文档拆分）：** https://github.com/robot-i-o/rio-hw
- **项目页：** https://robot-i-o.github.io/
- **文档站（MkDocs）：** https://robotio-docs.netlify.app/
- **论文：** https://arxiv.org/abs/2605.11564（RSS 2026 接收，项目页与 arXiv 摘要一致）
- **入库日期：** 2026-05-15
- **一句话说明：** 面向跨形态机器人学习的开源 Python **实时 I/O** 框架：以统一 **Node + 可插拔中间件** 抽象连接遥操作、传感、本体控制、RLDS 风格数据记录与 **异步策略推理**，换硬件组合以 **station 配置** 为主而非重写主循环；论文在单臂 / 双臂 / 人形与多种 VLA、DP、PPO 管线上给出系统验证。
- **沉淀到 wiki：** [RIO（Robot I/O）](../../wiki/entities/robot-io-rio.md)

---

## 文档 URL 说明

用户提供的 `https://robotio-docs.netlify.app/arXiv` 在 2026-05-15 抓取返回 **404**；官方文档入口以站点根路径 **https://robotio-docs.netlify.app/** 为准（含安装、Station 配置、Ubuntu RT 参考等）。论文全文仍以 arXiv PDF 为准。

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [VLA](../../wiki/methods/vla.md) | 论文主验证管线：π₀.5、GR00T N1.5 等 **微调 + 真机闭环**；强调 **action chunking** 与异步推理 |
| [LeRobot](../../wiki/entities/lerobot.md) | 文档叙述可 **导出到 LeRobot / DROID 格式** 对接常见训练栈；与 RIO 的「低延迟中间件直连推理」形成对照 |
| [Teleoperation](../../wiki/tasks/teleoperation.md) | 官方支持 Spacemouse、手柄、键盘、GELLO、手机、VR（Vision Pro / Quest 等）等多接口与滤波 |
| [Imitation Learning](../../wiki/methods/imitation-learning.md) | 示范数据经统一观测 schema 写入，下游 BC / VLA fine-tune |
| [Unitree G1](../../wiki/entities/unitree-g1.md) | Table II 列出的支持人形之一；另有 Booster T1、Franka / UR / xArm / SO-100 等 |

---

## 设计要点（论文 / 项目页归纳）

1. **Node 模板**：遥操作、传感器、机器人、策略等均以同一 Node 模式实现；**pub / req / pubreq** 三种循环语义，配合 **ring buffer**（固定频率状态流）与 **request queue**（异步命令）。
2. **中间件可切换**：同构 API 下可换 **Shared Memory / Thread / Portal / Zenoh / ZeroRpc** 等；共享内存侧重本机零拷贝，Zenoh/ZeroRPC 侧重跨机 TCP/IPC。
3. **Robot station**：单一 dataclass 配置声明拓扑（多臂、夹爪、腕部相机等），主循环对 **client API** 编程，隐藏进程与传输细节。
4. **数据**：RLDS 风格 step；**morphology** 过载各平台专有 state key；单位约定（米、弧度）；记录侧文档提到 **RoboDM** 压缩与可导出到各 dataloader。
5. **策略侧**：轻量 policy wrapper + **异步推理**；论文称通过中间件直连推理减少独立 policy server 的开销；项目页给出与 LeRobot 在相同硬件上的 **端到端观测–动作延迟** 对比量级（以官方 profiling 为准）。

---

## 对 wiki 的映射

- 新建 **`wiki/entities/robot-io-rio.md`**：跨形态实时控制与 VLA 部署工程实体页（架构 mermaid、与 LeRobot/VLA/遥操作互链）。
- 更新 `wiki/methods/vla.md`、`wiki/entities/lerobot.md`、`wiki/tasks/teleoperation.md`：补充交叉引用，避免孤岛页。

---

## 外部参考（便于复核）

- Ortega-Kral et al., *RIO: Flexible Real-Time Robot I/O for Cross-Embodiment Robot Learning*, arXiv:2605.11564
- [robot-i-o/rio（GitHub）](https://github.com/robot-i-o/rio)
- [RIO 项目主页](https://robot-i-o.github.io/)
- [RIO 文档站](https://robotio-docs.netlify.app/)
