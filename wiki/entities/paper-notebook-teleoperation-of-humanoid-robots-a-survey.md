---
type: entity
tags:
  - paper
  - survey
  - teleoperation
  - humanoid
  - shared-autonomy
  - human-robot-interaction
  - humanoid-paper-notebooks
  - iit
status: complete
updated: 2026-07-28
arxiv: "2301.04317"
related:
  - ../overview/paper-notebook-category-07-teleoperation.md
  - ../tasks/teleoperation.md
  - ../concepts/motion-retargeting.md
  - ../concepts/whole-body-control.md
  - ../comparisons/data-gloves-vs-vision-teleop.md
sources:
  - ../../sources/papers/humanoid_pnb_teleoperation-of-humanoid-robots-a-survey.md
  - ../../sources/sites/humanoid-teleoperation-survey.md
summary: "Teleoperation of Humanoid Robots（T-RO 2023）：以设备—重定向/规划—稳定器—WBC—关节控制为主链，系统整理反馈、共享自治、通信稳定、人因评价和应用；综述无配套实现代码。"
---

# Teleoperation of Humanoid Robots：人形遥操作综述

**Teleoperation of Humanoid Robots: A Survey**（[arXiv:2301.04317](https://arxiv.org/abs/2301.04317)，IEEE T-RO 2023，[网页版](https://humanoid-teleoperation.github.io/)）系统整理人形遥操作的输入/反馈设备、重定向、稳定与全身控制、共享自治、通信、人因评价和应用。

## 一句话定义

**这篇综述把人形遥操作抽象为“测人 → 重定向/规划 → 通信 → 稳定与全身控制 → 机器人执行 → 多模态反馈给人”的闭环，并强调接口、自治等级、延迟稳定性和人因指标必须联合设计。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 同时满足平衡、接触和操作任务的低层控制 |
| ZMP | Zero-Moment Point | 经典双足稳定与落脚规划判据 |
| DCM | Divergent Component of Motion | 连接质心动力学、平衡与遥操作同步的状态 |
| SA | Situational Awareness | 操作员对远端状态的感知、理解和预测 |
| SUS | System Usability Scale | 遥操作接口主观可用性量表 |
| NASA-TLX | NASA Task Load Index | 操作员多维工作负荷量表 |

## 为什么重要

- **人形不是双臂机械臂加摄像头：** 浮动基、欠驱动、混合接触和全身冗余使输入映射与稳定控制强耦合。
- **把“遥操作效果差”拆成可定位模块：** 传感、重定向、通信、稳定器、WBC、执行与反馈都可能是根因。
- **共享自治不是越高越好：** 高自治降低工作负荷却可能损伤态势感知；低自治提高控制权但增加认知与体力负担。
- **补足工程评价：** 只展示专家完成任务不足以证明系统可用，还要测效率、错误、SA、workload、presence 与 learnability。

## 核心信息

| 项 | 内容 |
|----|------|
| 发表 | IEEE Transactions on Robotics，2023 |
| 作者机构 | 意大利技术研究院（IIT）、IHMC、UIUC、AIST/CNRS/Inria 等 |
| 范围 | 输入设备、反馈、模型/重定向/规划、稳定与 WBC、共享自治、通信、评价、应用 |
| 双边核心矛盾 | 稳定性与透明度冲突；时延会向闭环注入能量 |
| 评价框架 | effectiveness、efficiency、satisfaction、SA、workload、presence |
| 开放状态 | **代码不适用**：综述与网页版公开，未提出配套实现系统 |

## 流程总览

```mermaid
flowchart LR
  human["操作员<br/>运动/力/生理状态"] --> sense["输入接口<br/>MoCap/Exo/VR/GUI"]
  sense --> retarget["重定向与规划<br/>共享自治"]
  retarget --> net["非理想通信<br/>延迟/丢包/带宽"]
  net --> stabilize["稳定器 + WBC<br/>关节控制"]
  stabilize --> robot["人形机器人<br/>远端环境"]
  robot --> feedback["视觉/听觉<br/>触觉/力反馈"]
  feedback --> human
```

## 核心机制（方法栈）

### 1）从人类意图到机器人参考

输入可从键鼠/GUI 的低频任务命令到 IMU、光学动捕、外骨骼的高频全身状态。重定向需处理人体与机器人尺寸、DoF、关节限位和接触差异；自治越低，输入频率与通信带宽要求通常越高。

### 2）稳定、全身控制与反馈

机器人侧依次用 ZMP、DCM、接触 wrench 等稳定表示生成质心/落脚/接触参考，再由 IK、逆动力学、动量或 QP WBC 分配到关节。视觉、力、触觉反馈提高远端感知，但过多、不同步或高延迟反馈也会增加 workload 与晕动。

### 3）延迟下的双边稳定

纯时延可能向闭环注入能量。波变量、passivity observer/controller 和 energy tank 可恢复无源性，但通常牺牲透明度。低延迟可直接闭环，中等延迟常用 move-and-wait，高延迟应提升本地自治或预测人/机器人状态。

### 4）人因与自治等级

评价至少同时测客观任务绩效与主观/生理信号。SUS 看可用性，SAGAT 看态势感知，NASA-TLX 看工作负荷；自治等级应围绕任务风险和失败模式调整，而非固定追求完全自动或完全手动。

## 源码运行时序图

**不适用。** 本文是文献综述，不提出单一可执行软件栈；官方网页版用于导航章节和参考文献，未列配套实现仓库。

## 工程实践与开源状态

| 设计问题 | 工程检查 |
|----------|----------|
| 输入 | DoF 覆盖、漂移/遮挡、标定、穿戴疲劳、采样率 |
| 映射 | 关节限位、构型连续、接触可行、失效回退 |
| 通信 | 单向/往返延迟、jitter、丢包、带宽和时钟同步 |
| 控制 | 平衡裕度、接触 wrench、饱和、急停与本地保护 |
| 反馈 | 视觉稳定、触觉有用性、不同通道时延一致性 |
| 评价 | task success/time/error + SUS/SAGAT/NASA-TLX + 新手学习曲线 |
| 开源状态 | **不适用**；综述没有官方运行代码，不应借第三方仓库冒充论文实现 |

## 与其他工作对比

| 控制形态 | 人的角色 | 优势 | 主要代价 |
|----------|----------|------|----------|
| 直接遥操作 | 连续给参考 | 控制权和临场性高 | 带宽/延迟敏感、负荷高 |
| 共享自治 | 给意图并监督 | 平衡绩效与负荷 | 权限切换、可预测性难 |
| 监督自治 | 下达任务/确认 | 容忍长延迟 | 本地感知规划要求高 |
| 双边遥操作 | 连续控制 + 力反馈 | 接触透明、可利用人运动技能 | 稳定—透明度冲突 |

## 实验与评测

本文是综述，不提供单一新系统或统一 benchmark 结果。其评测贡献是建立读法：

- **客观可用性：** 完成率、准确/完整性、任务时间、成本、能耗、错误和多余动作。
- **主观可用性：** SUS 与访谈，必须区分熟练专家和目标用户。
- **态势感知：** SAGAT 等 freeze-probe、眼动/EEG 或任务表现代理。
- **工作负荷：** NASA-TLX/SWAT 配合心率、眼动、呼吸等连续信号。
- **临场感：** presence/immersion 问卷与行为、生理反应。
- **文献缺口：** 系统性用户研究少，很多论文依赖单个专家演示。

## 结论

**人形遥操作的成败不是某个追踪器或控制器的单点指标，而是“人—接口—网络—全身动力学—反馈”闭环的联合可用性。**

1. **先按任务选自治等级** — 高风险/长延迟场景需要更多本地自治，精细接触仍需要人在环。
2. **稳定与透明不可同时无限提高** — 双边控制的无源化通常会增加阻尼、降低触感。
3. **延迟预算必须端到端测** — 传感、网络、规划、控制、显示不同步会叠加为晕动和失稳。
4. **评价必须包含普通用户** — 专家 demo 不能证明 learnability 与部署可用性。
5. **2023 后需结合新学习式系统更新阅读** — 综述不覆盖后来的大规模全身 tracking policy 与 XR 数据飞轮。

## 局限与风险

- 文献截点早于 2023 年，未覆盖 OmniH2O、TWIST2、BFM、现代 Vision Pro 遥操作与大规模示范数据。
- 分类跨度很大，但缺乏可复现的统一 benchmark 和汇总量化元分析。
- 许多被综述系统依赖专用人形、商业动捕或定制外骨骼，工程成本难横向归一。
- 将 passivity 作为稳定充分条件仍不保证高透明度或好任务表现。
- 人因评价建议完整，但文献本身也指出机器人论文常缺用户研究与生理/行为指标。

## 与其他页面的关系

- 任务主入口：[Teleoperation](../tasks/teleoperation.md)
- 映射：[Motion Retargeting](../concepts/motion-retargeting.md)
- 执行后端：[Whole-Body Control](../concepts/whole-body-control.md)
- 接口选型：[数据手套 vs 视觉遥操作](../comparisons/data-gloves-vs-vision-teleop.md)
- 路线位置：[遥操作纵深 Stage 0](../../roadmap/depth-teleoperation.md)

## 参考来源

- [humanoid_pnb_teleoperation-of-humanoid-robots-a-survey.md](../../sources/papers/humanoid_pnb_teleoperation-of-humanoid-robots-a-survey.md)
- [humanoid-teleoperation-survey.md](../../sources/sites/humanoid-teleoperation-survey.md)
- 论文：<https://arxiv.org/abs/2301.04317>
- IEEE DOI：<https://doi.org/10.1109/TRO.2023.3236952>

## 推荐继续阅读

- 官方网页版：<https://humanoid-teleoperation.github.io/>
- [遥操作纵深路线](../../roadmap/depth-teleoperation.md)
