# (4万字）Humanoid Hardware入门 101

> 来源归档（blog / 微信公众号）

- **标题：** (4万字）Humanoid Hardware入门 101
- **类型：** blog
- **作者：** human five（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/10hYwFzC1EuCypFVzC6QGQ
- **发表日期：** 2026-06-01
- **入库日期：** 2026-06-01
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0 + `wechat-article-for-ai`（Camoufox）；正文约 6.4 万字 / 78 图；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **原始落盘：** [wechat_humanoid_hardware_101_2026-06-01.md](../raw/wechat_humanoid_hardware_101_2026-06-01.md)
- **一句话说明：** 以第一性原理拆解人形机器人 **机身材料、执行器传动链、传感与手部、供配电与计算、BOM 成本与中美供应链**；核心判断：执行器（减速器+丝杠+轴承）是成本与规模化瓶颈，QDD 与高减速比路线之争由「可消耗件 + RL 扭矩透明」重塑，2 万美元 BOM 需架构简化而非仅压价。

## 核心摘录（归纳，非全文）

### 研究框架（引言）

- 回答三件事：子系统原理、工程设计权衡、BOM 成本与规模化压缩空间。
- 不预测「唯一最优设计」，而提供可迁移到任意人形/简化平台的 **故障点、制造难点、长期高成本部件** 分析框架。
- 供应链从 **产业密度、需求创造、资本配置** 三维对比中美。

### 七大子系统分组（对应 wiki 分类 hub）

| 组 | 公众号章节 | 分类 hub |
|----|------------|----------|
| **01 机身与材料** | 机身骨架 | [humanoid-hardware-101-chassis-materials](../../wiki/overview/humanoid-hardware-101-chassis-materials.md) |
| **02 关节传动与感知链** | 电机、减速器、编码器 | [humanoid-hardware-101-actuation-sensing-chain](../../wiki/overview/humanoid-hardware-101-actuation-sensing-chain.md) |
| **03 直线传动与轴承** | 丝杠、轴承 | [humanoid-hardware-101-linear-transmission-bearings](../../wiki/overview/humanoid-hardware-101-linear-transmission-bearings.md) |
| **04 集成执行器** | 执行器 | [humanoid-hardware-101-integrated-actuators](../../wiki/overview/humanoid-hardware-101-integrated-actuators.md) |
| **05 能源与计算电子** | 电池、计算单元、PCB | [humanoid-hardware-101-power-compute-electronics](../../wiki/overview/humanoid-hardware-101-power-compute-electronics.md) |
| **06 传感与末端** | 通用传感器、触觉、末端执行器 | [humanoid-hardware-101-sensing-end-effectors](../../wiki/overview/humanoid-hardware-101-sensing-end-effectors.md) |
| **07 产业与成本地缘** | 产业格局、成本分析、地缘格局、潜在领先者 | [humanoid-hardware-101-supply-chain-economics](../../wiki/overview/humanoid-hardware-101-supply-chain-economics.md) |

### 文内收束判断（策展）

- **执行器 BOM：** 宇树类关节约 2000 元/个、整机 25–30 个；减速器占执行器成本 45–50%，丝杠约占整机 BOM ~20%；行星滚柱丝杠为供应链瓶颈。
- **路线之争：** 谐波/RV 高减速比 vs 行星 QDD——后者 **扭矩透明** 更利 RL，前者利精密高载；减速器从「十年寿命精密件」转向 **12–18 个月可更换消耗件**。
- **电机：** 关节主流为 **有传感器电子换相、外转子、径向磁通、钕永磁**；灵巧手用 **空心杯**。
- **2 万美元 BOM：** 靠 **减关节数、模块复用、简化手部**，非单靠供应商降价；手部触觉可占单手 BOM 约 40%。
- **中美：** 中国优势在 EV/无人机/消费电子衍生的 **端到端供应链 + 产业密度**；美国资本偏 **软件/操作智能**，硬件 COGS 不确定抑制过早量产工厂投入。

## 对 wiki 的映射

| 主题 | 关系 |
|------|------|
| [Humanoid Hardware 101 技术地图](../../wiki/overview/humanoid-hardware-101-technology-map.md) | 父节点：七类子系统 + Mermaid 总览 |
| [人形硬件选型 Query](../../wiki/queries/humanoid-hardware-selection.md) | 整机平台选型；本专题补 **部件级** 权衡 |
| [开源人形硬件对比](../../wiki/entities/open-source-humanoid-hardware.md) | 研究向整机；与本文量产 BOM 视角互补 |
| [电机驱动与总线协议概览](../../wiki/overview/motor-drive-firmware-bus-protocols.md) | 执行器之上的固件/总线层 |
