# RoboParty Docs — Roboto Origin 基本介绍

> 来源归档（ingest · 产品文档站）

- **URL：** <https://roboparty.com/roboto_origin/doc>
- **标题：** 基本介绍 | RoboParty Docs
- **机构：** 机器人派对（RoboParty）
- **类型：** site（产品文档）
- **入库日期：** 2026-07-14
- **一句话说明：** Roboto Origin（萝博头原型机 / RPO）官方文档站入口：产品背景、参数表、系统架构与开源范围；与 GitHub [roboto_origin](../repos/roboto_origin.md) 聚合仓互为补充。

## 为什么值得保留

- **硬件参数一手表**：身高、DOF、重量、关节扭矩、电池与感知选项等数值以文档站为准，便于更新 [roboto-origin](../../wiki/entities/roboto-origin.md) 实体页。
- **全栈开源边界**：明确机械/电气/固件/训练/部署/URDF/知识库文档的开放范围，避免把聚合仓误当成 monorepo 开发主仓。
- **产品叙事**：记录「四个月完成能跑能跳原型机」与「Open Source, Open Future」定位，服务开源人形硬件对比页。

## 文档摘录（2026-07 抓取）

### 项目背景

- **中文名**：萝博头原型机；面向创客、科研与教育的全开源双足人形原型平台。
- **公司**：上海萝博派对科技有限公司，2025-02-21 成立；2025-04 正式投入人形研发，约四个月完成首台能跑能跳原型机。
- **开源动机**：记录早期探索，让社区基于真实硬件、代码与训练部署流程继续扩展。

### 产品定位

适用：教学实验、算法验证、运动控制研究、具身智能探索、工程训练、二次开发原型平台。

### 基本参数

| 项目 | 参数 |
|------|------|
| 形态 | 双足人形 |
| 身高 | 约 1.25 m |
| 自由度 | 23 DOF |
| 体重 | 约 34 kg |
| 大腿长度 | 250 mm |
| 小腿长度 | 300 mm |
| 单臂总长度 | 440 mm |
| 运动性能 | 最高约 3 m/s |
| 关节峰值扭矩 | 120 N·m（腿部）；手臂 27 N·m |
| 电池 | 48 V, 15 Ah |
| 续航 | 2 小时以上 |
| 可选感知 | Intel D435i 深度相机、3D 激光雷达 |

### 机械与系统特点

- 工业级 CNC + 并联脚踝 + 类人偏置关节；强调结构强度与运控基础。
- **模块化全链路开放**：硬件与电气、嵌入式、装配标定、算法仿真、部署 SDK、工业设计分模块组织。
- **实机友好**：部署与 SDK 面向真机运行（驱动、推理、手柄、Python 调用）。
- **可扩展**：ROS 2、URDF 与开放接口，可接感知、导航、遥操作与评测。

### 开源范围

| 范围 | 内容 |
|------|------|
| 机械与硬件 | 结构、PCB、制造资料与 BOM |
| 固件与镜像 | USB2CAN 固件、主控镜像构建 |
| 训练 | Isaac Lab、RSL-RL、AMP、BeyondMimic、Parkour、Sim2Sim 等 |
| 部署 | ROS 2 框架、驱动、推理、手柄、Python SDK |
| 机器人描述 | URDF、MJCF、仿真模型 |
| 知识库 | 安全、装配、走线、标定、部署、SDK、FAQ |

### 延伸链接（README_cn 交叉）

- 聚合仓：<https://github.com/Roboparty/roboto_origin>
- Know-How 飞书：<https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87>
- QQ 交流群：546376843

## 仓库命名说明（2026 README_cn）

官方子模块已从历史 **Atom01 / atom01_*** 更名为 **RPO / roboparty_*** 系列（快照目录 `modules/...` 仍可能保留兼容路径）。详见 [roboto_origin.md](../repos/roboto_origin.md) 更新表。

## 对 wiki 的映射

- 更新 [`wiki/entities/roboto-origin.md`](../../wiki/entities/roboto-origin.md)：补充参数表、开源范围、RPO 命名。
- 交叉：[`wiki/entities/open-source-humanoid-hardware.md`](../../wiki/entities/open-source-humanoid-hardware.md)、[`wiki/entities/roboparty.md`](../../wiki/entities/roboparty.md)
