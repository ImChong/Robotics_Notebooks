# COSA 0.5 发布：人形 VLA V³-0 的全身能力升级

> 来源归档（blog / 官方新闻）

- **标题：** COSA 0.5 发布：人形 VLA V³-0 的全身能力升级
- **类型：** blog
- **作者 / 组织：** 逐际动力（LimX Dynamics）
- **原始链接：** https://www.limxdynamics.com/en/news/BK000067
- **发表日期：** 2026-07-15
- **入库日期：** 2026-07-16
- **核对范围：** 2026-11-21 抓取并通读官方技术报告全文 `<https://limxdynamics.com/cosa05v3/>`（中英文 §1–§7 全部章节，含异步双 GPU 快慢 VLA V³-0、43-DoF Oli、Training-Time RTC、LimX-WBT vs SONIC 定量对比表、人类在环接管与真机 RL 实验结果及长程整理实录）以及官方新闻页 `<https://www.limxdynamics.com/zh/news/BK000067>` / `<https://www.limxdynamics.com/en/news/BK000067>`；COSA OS 本体与 V³-0 / LimX-WBT 未开源，无法核对内部代码实现。
- **通读：** 已完成
- **抓取方式：** 直接请求官方技术报告页 `https://limxdynamics.com/cosa05v3/` 与新闻页 `BK000067`（均返回完整静态/SSR HTML 文本），并与同期媒体报道及 [FluxVLA 文档站](https://fluxvla.limxdynamics.com/) 交叉核对
- **一句话说明：** 逐际动力发布 **LimX COSA 0.5** 技术报告与产品更新：在 **43-DoF LimX Oli** 上部署快慢异步 VLA **V³-0**（慢系统 `< 5 Hz` VLM + 快系统 `50 Hz` flow-matching + Training-Time RTC）与 **LimX-WBT**（S0 运控基础模型），并结合人类在环全身接管与真机 RL 闭环；同步开源 **Humanoid FluxVLA Engine**。

## 核心摘录（归纳，非全文）

### 发布要点

- **COSA（Cognitive OS of Agents）0.5**：逐际动力自研 **人形大脑操作系统**，强调「**模型是技能，系统才是大脑**」——与「单模型即大脑」或纯端到端 scaling 叙事形成对照。
- **V³-0**：本次升级重点在 **人形 VLA 的全身能力**——S1 层 VLA 需输出 **底盘 + 躯干 + 头 + 双臂 + 双手** 协同的 **全身动作块**，而非孤立机械臂轨迹；由 S0 **LimX WBT** 实时跟踪执行。
- **实机 Demo（Oli）**：晾衣、收纳玩偶、搬箱摞箱、深弯腰拾物、清理垃圾、递物等 **连续家务**；**任务间不剪辑、不复位**；媒体称与 Figure 并列为全球少数 **长程、无遥操、全尺寸人形家庭任务** 完整实录之一。
- **同步开源 [FluxVLA Engine](https://fluxvla.limxdynamics.com/)**：统一配置管理下训练/评测/推理 **π0、π0.5、GR00T、OpenVLA、LlavaVLA、DreamZero** 等；**首次支持端侧（本机）推理**。

### 三层架构（COSA 0.5）

| 层 | 名称 | 频率（约） | 职责 |
|----|------|------------|------|
| **S2** | 认知层 | ~1 Hz | 头部/手腕相机 + 语言；场景理解、记忆、推理、**任务调度**（「做什么」） |
| **S1** | 技能层 | ~50 Hz | **可复用技能集合**（VLA、WAM 等）；多元数据、**按技能独立迭代**（「用什么技能做」） |
| **S0** | 运控层 | 1000 Hz | **LimX WBT** 全身运动基础模型（~千万参数 Transformer，**完全机上**）；统一接口接收全身运动目标 → 平衡协调关节指令 |

**链路示例（收拾衣服）：** S2 理解「捡椅子衣服、取衣架衣服、扔进脏衣篓」→ S1-VLA 生成接近、站位、弯腰、抓取、起身、移动等 **全身动作序列** → S0-WBT 在重心变化中维持平衡并精准跟踪。

### LimX WBT（S0）要点

- **一次训练、三类复用：** VLA 执行、遥操作采集、零样本回放。
- **公开对标 SONIC（行业公开最强 WBT 之一）：** 平均关节角误差 **3.3° → 1.5°**；MPJPE **13.75 mm → 12.85 mm**；动作平滑度指标约 **低 11% / 20%**（越低越好，以官方发布材料为准）。
- **与 SONIC / Any2Any 关系：** LimX 自研 WBT 栈；[Any2Any](../../wiki/entities/paper-any2any-cross-embodiment-wbt.md) 曾以 Gear-SONIC 为源骨干做跨机体迁移——COSA 侧强调 **软硬一体 + 机上实时** 的产品闭环。

### 行业定位（发布材料语境）

| 维度 | Figure（Helix 02） | Flexion（Reflect v1.0） | 逐际动力（COSA 0.5） |
|------|-------------------|-------------------------|----------------------|
| 分层 | System 2 / 1 / 0 | Command / Motion / Control | S2 / S1 / S0 |
| 哲学 | 统一神经系统、端到端连接 | 分层自主栈（第三方 G1 硬件 Demo） | **OS 调度多模型/技能**；自研 Oli 本体 |
| 长程 Demo | 厨房/卧室整理（~4 min 级） | 户外拾取、搬运（复杂度叙事较低） | 一镜到底客厅家务 |
| 开放 | 全栈闭源 | — | **FluxVLA Engine 开源** |

## 对 wiki 的映射

| 目标 | 说明 |
|------|------|
| [limx-cosa](../../wiki/entities/limx-cosa.md) | **新建** — COSA 人形大脑 OS 实体页（三层架构 Mermaid、与 Figure/Flexion 对照） |
| [fluxvla-engine](../../wiki/entities/fluxvla-engine.md) | **新建** — FluxVLA 开源工程底座实体页 |
| [vla](../../wiki/methods/vla.md) | V³-0 作为 S1 技能层人形 VLA 实例；FluxVLA 工程入口 |
| [loco-manipulation](../../wiki/tasks/loco-manipulation.md) | 长程全身移动操作产业 Demo 索引 |
| [whole-body-tracking-pipeline](../../wiki/concepts/whole-body-tracking-pipeline.md) | LimX WBT 作为 S0 产品化 WBT 锚点 |
| [sonic-motion-tracking](../../wiki/entities/nvidia-sonic.md) | WBT 对标参照 |
| [notable-commercial-robot-platforms](../../wiki/overview/notable-commercial-robot-platforms.md) | LimX 平台纵览补链 |

## 可信度与使用边界

- 本文为 **官方产品发布 + 媒体解读** 归纳；技术细节以 [limxdynamics.com](https://www.limxdynamics.com)、[fluxvla.limxdynamics.com](https://fluxvla.limxdynamics.com/) 与后续论文/代码为准。
- **V³-0** 命名来自官方新闻标题；公开材料对 S1 层多称「VLA 技能」，未单独披露 V³-0 网络结构——wiki 页保持 **系统层 + 能力边界** 归纳，不臆造架构细节。
- 一镜到底 Demo **≠** 商业化成功率；长期稳定性、异常恢复与规模部署仍待验证。

## 推荐外部链接

- [官方新闻（EN）](https://www.limxdynamics.com/en/news/BK000067)
- [FluxVLA Engine 文档](https://fluxvla.limxdynamics.com/)
- [FluxVLA Engine 产品页](http://fluxvla.limxdynamics.com/en/)
