# Solving Dexterity: A Full-Stack Approach — mimic hand M1 & mimic wearable U1

> 来源归档（blog / mimic robotics 官方）

- **标题：** Solving Dexterity: A Full-Stack Approach
- **类型：** blog
- **作者：** mimic robotics
- **原始链接：** https://www.mimicrobotics.com/blog/solving-dexterity-a-full-stack-approach
- **发表日期：** 2026-07-16
- **入库日期：** 2026-07-17
- **抓取方式：** 官方博客页直接抓取（WebFetch + HTML 链接核查）
- **一句话说明：** mimic 宣布 **mimic hand M1**（高度后驱、腱驱动、瑞士自研工业灵巧手）与 **mimic wearable U1（umimic）**（与 M1 运动学/传感/视觉 1:1 对应的被动外骨骼示范采集器），并公开 **全栈 Physical AI** 叙事：以人手为固定形态贯穿预训练—可穿戴中层数据—真机遥操作/部署；配套自研 **mimic-ipc** 零拷贝中间件、统一遥测与 **Smooth Operator（SBR）** 采样式重定向快照。

## 核心摘录（归纳，非全文）

### 全栈赌注与数据金字塔

- **唯一规模化路径：** 在 Physical AI 前沿 **垂直整合全栈**（类比自动驾驶），硬件需 **完全可观测**。
- **跨形态鸿沟：** 用人视频预训练、二指夹爪后训练会引入 **pre-training / post-training 不对齐**；mimic 赌注是 **全链路固定「人手」形态**。
- **数据金字塔（质量 × 规模）：**
  - **底层：** 人类视频（单样本质量低、规模大）
  - **中层：** 可穿戴设备数据（质量高于任意 egocentric 视频、比真机遥操作更易扩展）
  - **顶层：** 真机遥操作与部署数据（质量最高、成本最高）

### mimic hand M1（硬件）

| 维度 | 规格（博客表） |
|------|----------------|
| 制造 | 瑞士自研自产 |
| 驱动 | **双向、滑轮导向腱驱动**（非 Bowden 管；轴承/滑轮降摩擦） |
| DoF | **15 主动 + 6 耦合 = 21** |
| 后驱性 | 反驱扭矩 **< 0.05 Nm**；可经电机电流感知 **50 g** 级载荷 |
| 力估计 | 双编码器（电机+关节）；单向接触力 **< 0.1 N** |
| 指尖力 | 稳态 **25 N**（伸直） |
| 稳态载荷 | 圆柱强力抓握 **> 25 kg** |
| 指尖定位（闭环） | **± 0.18 mm** |
| 关节背隙 | **< 0.3°** |
| 触觉 | 法向力、切向剪切力、多点接触位置 |
| 相机 | 全局快门、同步 |
| 设计哲学 | **AI-first**：按工业任务数据裁剪 DoF；指尖区保人形几何，腕部加宽以容纳线性腱路由 |

**架构取舍：** 掌内集成电机手 vs **前臂腱驱动**——后者用前臂体积换 **持续扭矩 + 后驱性 + 工业级执行器供应链**；避免 Bowden 摩擦随腕角/循环漂移。

### mimic wearable U1 / umimic（数据采集）

- **定位：** 扩展 **UMI** 思路到 **灵巧手**：操作者经外骨骼直接动指，而非握二指夹爪或遥操作真机。
- **1:1 对应：** 刚性连杆 **机械约束** 到人手可做且 M1 可做的运动；**14 跟踪 + 6 耦合 = 20 DoF**；拇指 CMC 对位最难（四指在机器人指耦合后、拇指在上方）。
- **传感对齐：** 指尖触觉、关节编码器、腕相机与 M1 **一一对应** → 示范与机器人观测模态一致。
- **取舍：** **固定几何、适配手寸范围**（非每手自适应），换取 **无延迟、无重定向、近原生速度** 与更忠实示范。

### 基础设施（软件）

- **mimic-ipc：** 硬实时零拷贝 IPC；相对 ROS2 FastDDS shared memory，HD 图像负载作者称 **~18 000×** 加速、中位延迟 **89 ns**（跨核）；232 B 控制话题抖动 **~350×** 低于 FastDDS。
- **遥操作栈：** 15 DoF 手需低抖动重定向；发布 **Smooth Operator（SBR）** 采样式重定向快照 → [项目页](https://mimicrobotics.github.io/smooth-operator/) / [代码](https://github.com/mimicrobotics/mimic_retargeter_lab/) / [arXiv:2607.07491](https://arxiv.org/abs/2607.07491)。
- **遥测：** 复用 mimic-ipc 零拷贝；持久化异步化；观测与数据采集 **统一视图**，支持机队级预测性维护。

### 与 AI 路线关系

- 硬件/数据/中间件为 **[mimic-video](../../wiki/methods/mimic-video.md)（VAM）** 与后续 **Video Action Models** 产业化铺路；博客称将很快发布下一步更新。

## 对 wiki 的映射

- [mimic hand M1](../../wiki/entities/mimic-hand-m1.md) — 产业灵巧手实体
- [mimic wearable U1](../../wiki/entities/mimic-wearable-u1.md) — 可穿戴示范采集实体
- [mimic-video（VAM）](../../wiki/methods/mimic-video.md) — 同公司视频-动作模型路线
- [Teleoperation](../../wiki/tasks/teleoperation.md) — U1 中层数据与真机遥操作对照
- [灵巧操作数据采集指南](../../wiki/queries/dexterous-data-collection-guide.md) — 固定运动学外骨骼 vs 视觉/手套方案
