# General Manipulation Model — Introducing the VLOA Large Model (Part 2)

> 来源归档（blog / RoboScience 官方英文站）

- **标题：** General Manipulation Model — Introducing the VLOA Large Model (Part 2)
- **类型：** blog
- **作者：** RoboScience
- **原始链接：** <https://en.roboscience.co/jishu_detail/20.html>
- **发表日期：** 2026-05-19
- **入库日期：** 2026-07-19
- **抓取方式：** 官方技术博文页面直接抓取（WebFetch）
- **一句话说明：** VLOA 第二引擎 **通用操作模型**：接收世界模型 **Object Trajectory**（物体与环境 3D 点云序列），以 **轨迹条件** 学习「如何到达」而非重学「去哪」，>1B 参数跨技能联合训练，>3 fps 闭环输出关节角；强调跨本体灵巧手（X-hand / LEAP Hand）、细粒度力控与长程多步任务。

## 核心摘录（归纳，非全文）

### 问题重框

- 操纵模型三大瓶颈：**新物体泛化差**、**细粒度力控难**、**长程误差累积**。
- 常见 **原子技能库**（抓取/放置分立子模型）扩展性差。
- RoboScience：**>1B 参数统一操作表示**，共享物理常识与轨迹先验，跨物体/任务/平台。

### 与世界模型的接口

| 要素 | 内容 |
|------|------|
| **输入** | Embodied World Model 的 **Object Trajectory**（物体 + 环境点云，含位姿与形变） |
| **学习目标** | **轨迹条件**：轨迹已含几何/物理先验，模型学接触点、力控与关节指令 |
| **推理** | **>3 fps**；点云输入闭环控制关节角 |
| **闭环工作流** | **物理引擎 → 仿真数据 → 端到端训练** |

### 三项演示亮点（博文）

1. **任意物体抓取** — 桌面与收纳盒内杂乱场景；跨 **X-hand**（12 DoF、80 N 夹持力）与 **LEAP Hand**（16–20 DoF）无需改策略。
2. **细粒度操作** — 立硬币、开信封、夹薯片、注射器定量注液、夹海苔/蛋壳/甜筒等毫牛级力控。
3. **长程与闭环** — 家具组装（读说明书、双臂协作、力反馈恢复）；传送带动态抓取。

### 四项技术属性

- **全空间物体** — 刚体、铰接体、1D/2D/3D 可形变物体。
- **跨本体闭环** — 臂/人形/灵巧手；视觉+触觉+力多模态在线调整。
- **物理仿真闭环** — 大规模仿真预训练 + 少量真机微调。
- **Scaling** — 100 亿条仿真操纵数据上成功率呈幂律提升；200M 规模模型随抓取样本增加成功率与关节角方差（多样性）同步上升。

### VLOA 完整闭环（结论段）

- **Embodied World Model**：3D 点轨迹想象；>100 万小时视频数据。
- **General Manipulation Model**：轨迹 → 接触/力/关节；100 亿条仿真操纵数据。
- **Object Trajectory Interface** 串联感知到执行。

### 对 wiki 的映射

- 实体页：[wiki/entities/roboscience-vloa.md](../../wiki/entities/roboscience-vloa.md)
- 任务：[wiki/tasks/manipulation.md](../../wiki/tasks/manipulation.md)
- 方法：[wiki/methods/vla.md](../../wiki/methods/vla.md)（VLOA 命名与级联对照）
- 可微仿真：[wiki/entities/newton-physics.md](../../wiki/entities/newton-physics.md)、[wiki/entities/brax.md](../../wiki/entities/brax.md)
