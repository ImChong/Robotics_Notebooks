# RoboFinals：首个 Newton-Native 全栈评测基准

> 来源归档（site / 媒体文）

- **标题：** The First Newton-Native Benchmark: Running the Full Evaluation Stack on Newton
- **类型：** site（厂商媒体发布）
- **来源：** 光轮科技（Lightwheel）
- **链接：** https://lightwheel.ai/media/lightwheel-launches-robofinals-full-stack-robot-evaluation-benchmark-newton
- **入库日期：** 2026-09-14
- **一句话说明：** RoboFinals 在开源 Isaac Lab-Arena 上跑通 **Newton 全栈评测**：资产/求解器/机器人/遥操作数据/训练与评测均在 Newton 原生构建；首发 **22** 个家庭/医院/工厂接触丰富任务；光轮任 Newton TSC 成员并主导可变形求解器。
- **代码：** RoboFinals 平台仍 **商业闭源**；[Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena)（Apache 2.0）已开源
- **沉淀到 wiki：** [`wiki/entities/lightwheel-robofinals.md`](../../wiki/entities/lightwheel-robofinals.md)

---

## 步骤 2.5（开源核查，2026-09-14）

| 项 | 结论 |
|----|------|
| Newton-native 22 任务 benchmark | **商业平台内**；需 Book a Demo，无公开任务包下载 |
| 遥操作数据集 | 文内称每任务数百条 demo + 质检；**未列公开 HF/GitHub** |
| Isaac Lab-Arena | **已开源** — `isaac-sim/IsaacLab-Arena` |
| Newton 引擎 | **已开源** — `newton-physics/newton`（Linux Foundation） |

## 官方要点摘录

### 定位

- 首个 **fully Newton-native** 评测 benchmark：不仅是「换物理后端」，而是 **每一层**（资产、求解器、机器人、遥操作、数据、评测）在 Newton 上重建并端到端验证。
- 叙事：分数可信度取决于物理引擎；Newton 需要标准资产、任务与计分协议才能成为 **共享 ground truth**。

### 全栈分层（Newton 上重建）

| 层 | 要点 |
|----|------|
| **资产与环境** | 按最新 Newton schema 建 SimReady 资产；刚体/铰接/线缆等可变形体；参数来自 **实测标定**；入库前在引擎内交互验证 |
| **物理与求解器** | Warp + Newton 刚体；光轮扩展 **线缆/布料** 可变形求解器、多物理耦合与无穿透碰撞 |
| **机器人** | 四类具身验证：**X7S、Dexmate、H2+、G1** |
| **遥操作** | Newton 原生环境内采集 demo，避免跨环境轨迹导入 |
| **数据** | 全任务数百条/任务 teleop；标准化质检 |
| **评测** | 仍在 **Isaac Lab-Arena** 跑 rollout；每任务成功判据 + 协议 + 计分脚本 |

### Newton-native 首发任务集（22）

| 场景 | 示例任务 |
|------|----------|
| **工厂** | 连接器插入对齐、剔除不良件、不规则姿态拣选、线缆插拔确认、托盘拣选 |
| **家庭** | 开柜门/抽屉、冰箱取物、装洗碗机、微波炉门与按钮、整理台面 |
| **医院** | 无菌室手术器械整理、铰接器械（钳/剪）入托盘槽与 stringer |

强调：**接触丰富、长时域** —— 摩擦、铰接约束、变形与持续接触。

### 闭环验证

- 管线：Newton 内 teleop 采集 → Isaac Lab 上训练 → RoboFinals 标准化任务评测。
- **局限自述：** 训练与评测均在仿真内，**不单独证明真机一致**；真机相关性依赖上游实测资产与接触/变形求解器，**计划后续发布** Sim–Real 相关结果。

### 光轮与 Newton 关系

- Newton 由 NVIDIA、Google DeepMind、Disney Research 发起。
- 光轮在 **Technical Steering Committee**，制定资产标准并主导可变形求解器开发。

## 对 wiki 的映射

- 实体页 → [`wiki/entities/lightwheel-robofinals.md`](../../wiki/entities/lightwheel-robofinals.md)
- Newton 引擎 → [`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md)
- Arena 底座 → [`wiki/entities/isaac-lab-arena.md`](../../wiki/entities/isaac-lab-arena.md)
