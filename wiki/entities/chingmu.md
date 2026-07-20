---
type: entity
tags: [industry, motion-capture, mocap, humanoid, dataset, teleoperation, chingmu, embodied-ai]
status: complete
updated: 2026-07-20
related:
  - ../concepts/motion-retargeting.md
  - ../tasks/teleoperation.md
  - ./freemocap.md
  - ./motioncode.md
  - ./unitree-g1.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/sites/chingmu.md
  - ../../sources/repos/cmrobot-motiondecode.md
  - ../../sources/repos/chingmu-github-sdks.md
summary: "青瞳视觉（CHINGMU）：自研光学动捕全栈厂商，面向人形/具身场景提供评测、遥操作、DAQ 采集场与 MotionDecode 千小时运动数据；核心软件商业授权，SDK 插件与 HF 数据集部分开放。"
---

# 青瞳视觉（CHINGMU Vision）

**青瞳视觉**（[en.chingmu.com](https://en.chingmu.com/)，上海青瞳视觉科技有限公司）是国内 **光学动作捕捉（MoCap）全栈自研** 供应商：自 **2015** 年起覆盖相机硬件、实时求解软件与行业方案，并将 **具身智能 / 人形机器人** 列为核心应用之一——从 **运动真值评测**、**遥操作映射** 到 **多模态 DAQ 数据采集场**，并运营 **MotionDecode** 千小时人体运动数据计划。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MoCap | Motion Capture | 光学/惯性等方式记录三维运动，常作机器人 IL/评测真值 |
| DAQ | Data Acquisition | 数据采集；官网指多模态具身训练场方案 |
| 6DoF | Six Degrees of Freedom | 三维位置 + 三维姿态的刚体位姿描述 |
| Retargeting | Motion Retargeting | 将人体轨迹映射到目标机器人骨架 |
| IL | Imitation Learning | 模仿学习；MotionDecode 宣称服务此下游 |
| ROS | Robot Operating System | 官网称性能评测方案可对接 ROS/ROS2 |

## 为什么重要

- **产业侧 MoCap 锚点：** 与 [FreeMoCap](./freemocap.md)（开源多相机）、[MotionCode](./motioncode.md)（运动数据公司）并列，代表 **商业光学动捕 + 具身数据** 路线；适合在选型时对照「自建棚 vs 采购全栈 vs 直接下载数据集」。
- **机器人四条产品线清晰：** 官网将 Robotics 拆为 **性能评测、灵巧手、遥操作、DAQ 采集场**，分别对应 **真值标定、手–物 IL、全身遥操、多模态训练集** 四条常见工程需求（见 [Teleoperation](../tasks/teleoperation.md)）。
- **公开数据资产：** [MotionDecode](https://huggingface.co/datasets/CMRobot/MotionDecode) 在 Hugging Face 提供 **样例 + G1 重定向 CSV**，降低「有 MoCap 叙事但缺可下载数据」的摩擦；完整 **1000h** 仍需申请。

## 核心结构（硬件 + 软件 + 数据）

| 层级 | 代表 | 角色 |
|------|------|------|
| **硬件** | K / MC / R 系列相机、PULSEH 光惯手套 | 亚毫米级光学跟踪；手套补手指/手腕 |
| **软件** | CMAvatar、CMTracker、CMVision、SDK | 实时 6DoF 求解、可视化、同步与导出 |
| **机器人方案** | Performance eval / Dexterous Hand / Teleop / DAQ | 评测真值、手–机 IL、低时延映射、多模态采集场 |
| **数据计划** | MotionDecode | 1000h+ 人体运动；BVH/CSV/视频/物体 6D/标签 |

## 流程总览（具身 DAQ → 训练）

```mermaid
flowchart LR
  A["表演者 + 道具"] --> B["光学相机阵列<br/>K / MC / R + 手套"]
  B --> C["CMAvatar 实时求解<br/>6DoF + 多模态同步"]
  C --> D["导出 BVH / CSV / 视频 / 物体 6D"]
  D --> E["重定向<br/>例：Unitree G1 CSV"]
  E --> F["下游 IL / RL / Sim 验证<br/>ROS · Python · MuJoCo 等"]
```

## 工程实践

| 场景 | 官网主张 | 工程注意 |
|------|----------|----------|
| **人形性能评测** | 轨迹偏差、重复性、多机协同误差一键报告 | 动捕提供 **全局真值**；仍需对齐机器人本体坐标与关节定义 |
| **遥操作** | 亚毫米映射 + ms 级闭环 | 与 VR/手柄路线互补；磁干扰场景可依赖光惯融合手套 |
| **灵巧手 IL** | 人/机双手 6DoF 同步 | 下游仍需接触相位与动力学可行性筛选 |
| **数据集** | HF `CMRobot/MotionDecode` 样例；100h G1 CSV | 引用需注明 Chingmu；跨机型迁移见 [Motion Retargeting](../concepts/motion-retargeting.md) |

### 开源状态（项目页核查，2026-07-20）

| 资产 | 状态 |
|------|------|
| **CMAvatar 等核心软件** | **商业软件**，官网 Software download；**非开源** |
| **引擎 SDK** | **部分开源** — [ChingMuVisionTech](https://github.com/ChingMuVisionTech)（UE/Unity/C++/Python/iClone 等插件） |
| **MotionDecode 数据** | **部分开源** — HF 公开样例与元数据；完整库 **申请 / Request access**（[申请表](https://v.wjx.cn/vm/rqsTPkU.aspx)，**MotionDecode@chingmu.com**） |
| **策略训练代码** | **未列** |

## 常见误区或局限

- **营销页 ≠ 学术 benchmark：** 「全球领先」「百人纪录」等属品牌与工程展示，**不能替代** 第三方对标实验；与 Vicon / OptiTrack 的精度对比需看具体场景与标定。
- **数据集开放 ≠ 软件开源：** MotionDecode 样例可下，但 **求解器与采集软件仍属商业产品**。
- **G1 重定向 CSV ≠ 即插即用策略：** 仍须处理接触、执行器限制与 Sim2Real；见 [Unitree G1](./unitree-g1.md) 与重定向概念页。
- **HF 与新闻表述差异：** 新闻写「1000h 免费申请」，HF 页同时提供 Request access 与 Discord；以仓库 README 与申请回执为准。

## 关联页面

- [Motion Retargeting（动作重定向）](../concepts/motion-retargeting.md) — MotionDecode 与 DAQ 下游关键步骤
- [Teleoperation（遥操作）](../tasks/teleoperation.md) — 动捕驱动全身遥操与数据采集
- [FreeMoCap](./freemocap.md) — 开源低成本 MoCap 对照
- [MotionCode](./motioncode.md) — 另一产业侧运动数据供应商叙事
- [Unitree G1](./unitree-g1.md) — MotionDecode 预重定向目标平台之一

## 参考来源

- [青瞳视觉官网归档](../../sources/sites/chingmu.md)
- [CMRobot/MotionDecode 数据集](../../sources/repos/cmrobot-motiondecode.md)
- [ChingMu GitHub SDK 组织](../../sources/repos/chingmu-github-sdks.md)

## 推荐继续阅读

- [CHINGMU 英文官网 — Robotics 应用](https://en.chingmu.com/list/46.html)
- [MotionDecode 数据集（Hugging Face）](https://huggingface.co/datasets/CMRobot/MotionDecode)
- [MotionDecode 开放计划新闻](https://en.chingmu.com/company-news/10746.shtml)
