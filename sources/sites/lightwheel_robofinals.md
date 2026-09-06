# Lightwheel RoboFinals（发布页）

> 来源归档

- **标题：** Lightwheel Unveils RoboFinals
- **类型：** site（厂商产品发布 / benchmark 平台）
- **来源：** 光轮科技（Lightwheel）
- **链接：** https://lightwheel.ai/robofinals
- **机构站：** https://lightwheel.ai/
- **发布日期：** 2025-12-04
- **入库日期：** 2026-09-06
- **一句话说明：** 光轮 **RoboFinals**：面向 VLA/通才机器人基础模型的工业级仿真评测平台；核心 **RoboFinals-100**（100 任务）基于 SimReady 资产；底座为 NVIDIA Isaac Lab-Arena；**Coming soon**，需预约 Demo。
- **代码：** 无独立公开仓库（平台为商业服务）；底层 [Isaac Lab-Arena](https://github.com/NVIDIA/IsaacLab-Arena)（Apache 2.0）、[LW-BenchHub](https://github.com/LightwheelAI/LW-BenchHub)（Apache 2.0）已开源
- **沉淀到 wiki：** [`wiki/entities/lightwheel-robofinals.md`](../../wiki/entities/lightwheel-robofinals.md)

---

## 步骤 2.5（开源核查）

| 项 | 结论 |
|----|------|
| RoboFinals 平台 | **商业闭源服务**（Book a Demo）；截至入库日标注 **Coming soon** |
| RoboFinals-100 任务/资产 | 未公开完整任务列表与权重；基于 Lightwheel **SimReady** 资产生态 |
| 评测底座 | **已开源** — [NVIDIA/IsaacLab-Arena](https://github.com/NVIDIA/IsaacLab-Arena)；与光轮联合设计评测与任务层 |
| 场景库 | **已开源** — [LightwheelAI/LW-BenchHub](https://github.com/LightwheelAI/LW-BenchHub)（RoboCasa/LIBERO 等 138+ 任务） |
| AutoDataGen | 媒体文介绍为 Isaac Lab 附加包；**未列公开 GitHub**（截至 2026-09-06） |

## 官方要点摘录

### 定位

- 业界首个「**足够难**、工业级、可承载前沿基础模型」的仿真评测平台，面向 **Frontier Labs** 的 VLA/通才模型。
- 痛点：学术 benchmark 已被前沿模型刷满，真机评测无「shadow mode」、成本高、难扩展；现有仿真任务过简或与真实脱节。

### RoboFinals-100

- **100 任务**，基于 Lightwheel **SimReady Asset** 标准。
- **领域：** 家庭（清洁/整理/收纳/摆放）、工厂（搬运/装配/机台交互）、零售（补货/分拣/货架）。
- **交互覆盖：** 刚体、铰接体（家电/柜门/旋钮）、可变形体（线缆/布料/液体）。
- **跨具身：** 桌面臂、移动操作、全身 loco-manipulation 三类。
- **统一成功判据**，支持跨团队公平对比。

### 平台能力

- 基于 **NVIDIA Isaac Lab-Arena**（与 NVIDIA 联合开发）。
- **大规模批量评测**：自动执行、日志、按任务类型/难度/领域聚合指标。
- **部署：** 云 API（快速迭代）或 **on-premise**（数据与流程自控）。
- **多物理后端：** Isaac Lab + **Newton**（主工业求解器）、Isaac Lab + PhysX、**MuJoCo**、**Genesis**；汇总为统一记分板。

### Real2Sim / Sim2Real

- SimReady 库全链路 **Real2Sim 标定**。
- 在建 **受控真机 benchmark** 与 **Sim–Real 相关性数据集**，验证前沿 VLA 迁移性。

### 合作

- **Qwen Team** 共同定义工业场景与评测标准；使用 RoboFinals 做高吞吐行业对齐评测。

## 对 wiki 的映射

- 实体页 → [`wiki/entities/lightwheel-robofinals.md`](../../wiki/entities/lightwheel-robofinals.md)
- 评测底座 → [`wiki/entities/isaac-lab-arena.md`](../../wiki/entities/isaac-lab-arena.md)
- 场景库 → [`wiki/entities/lw-benchhub-tour.md`](../../wiki/entities/lw-benchhub-tour.md)
- 工业灵巧规格对照 → [`wiki/entities/dexbench.md`](../../wiki/entities/dexbench.md)
