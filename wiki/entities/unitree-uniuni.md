---
type: entity
project_id: unitree-uniuni
project: https://uniuni.unitree.com/
tags: [software, multi-robot, humanoid, unitree, choreography, localization]
status: complete
updated: 2026-10-20
related:
  - ./unitree.md
  - ./unitree-g1.md
  - ./unitree-unistore.md
  - ../concepts/humanoid-multi-robot-coordination.md
  - ../overview/navigation-slam-autonomy-stack.md
sources:
  - ../../sources/sites/unitree-uniuni.md
summary: "UniUni 是宇树面向机器人编队演出的闭源桌面系统：用 2D/3D 轨迹、时间线、点云定位和动作市场组织多机表演，AI 辅助生成编排后仍可人工微调。"
---

# UniUni（宇树智能集群控制系统）

**UniUni** 是宇树科技面向多机器人编队与舞台演出的桌面应用，官网提供 Windows x64 与 Apple Silicon macOS 下载。它把轨迹设计、机器人定位、动作编排和现场监控放在同一可视化流程中。

## 一句话定义

UniUni 是面向宇树机器人表演编队的**图形化任务编排与同步执行工具**，覆盖地图定位、路径布置、动作时间线和运行监控。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SLAM | Simultaneous Localization and Mapping | 同步定位与建图；官网描述激光点云建图和定位流程 |
| API | Application Programming Interface | 时间线用于绑定动作 API；页面未公开接口规范 |
| AI | Artificial Intelligence | 官网称可按自然语言指令生成轨迹与动作编排建议 |
| 2D / 3D | Two-Dimensional / Three-Dimensional | 分别用于平面路径编辑与空间姿态预览 |

## 为什么重要

- **降低群体演出编排门槛：** 轨迹、队形和动作可通过图形界面编辑，不要求使用者逐台手工编程。
- **把空间定位纳入演出工作流：** 点云地图、坐标转换、初始位姿和阵型布置与动作时间线衔接，适合分析表演机器人从场地准备到同步执行的系统需求。
- **与算法研究边界不同：** 官网展示的是产品化工作流，未披露定位算法、通信同步策略、运动控制器或失败恢复机制；不应将演出效果等同于公开可复现的群体控制方法。

## 核心功能

| 功能 | 工具提供的操作 | 工程关注点 |
|------|----------------|------------|
| 轨迹编辑 | 2D/3D 双视图，贝塞尔曲线绘制每台机器人路径 | 路径约束、机器人间距与轨迹时间参数化未公开 |
| 时间线编排 | 拖拽动作 API，支持音乐伴舞、调试和循环模式 | 动作时长、过渡曲线与多机时钟同步细节未公开 |
| 地图与定位 | 查看激光点云、管理地图、定位、坐标转换、定位测试 | 官网未给定位误差、更新频率及遮挡鲁棒性指标 |
| 阵型规划 | N×M 阵型、间距/偏移批量规划和碰撞检测 | 需在真实场地验证机器人外形包络、制动距离和安全边界 |
| 运行监控 | 显示在线状态、定位、温度和雷达异常，提供急停 | 急停链路、失联策略和故障安全等级未披露 |
| AI 辅助编舞 | 按文字指令生成轨迹与动作序列，再同步到时间线调整 | 生成结果应人工检查路径、碰撞和动作兼容性 |

## 工程实践

官网给出的典型流程是：

1. 安装 Windows 或 macOS 客户端并登录宇树账号，让客户端发现局域网设备。
2. 导入点云地图，完成建图/定位准备，规划机器人阵型与行走轨迹。
3. 在时间线上为机器人绑定动作并配置演出模式。
4. 将机器人置于初始位置，初始化定位并同步地图、轨迹和任务。
5. 启动演出，监控定位、温度、雷达与在线状态，必要时触发急停。

**选型与验证建议：** 在正式演出前，用目标机型和真实场地验证定位重定位、无线网络拥塞、动作时序偏差、避碰余量、断连恢复及急停响应；这些指标没有在公开产品页给出。

## 局限与风险

- **开源状态：闭源软件。** 截至 2026-10-20，官网提供 v1.0.1 安装包，但没有列出 GitHub 或其他公开源码地址。
- 官网没有列出完整机型/固件兼容矩阵、可编程接口文档、集群规模上限、定位精度、同步误差或网络要求；这些需要向厂商确认并在目标部署环境实测。
- 页面中春晚、赛事和活动案例属于厂商陈述，本页未独立核验，也不能据此推断任意规模或场地都能复现相同效果。
- AI 自动编排不能替代碰撞检查和现场安全流程，尤其要审查路径交叉、机器人间距与急停覆盖。

## 与其他页面的关系

- [人形多机协调](../concepts/humanoid-multi-robot-coordination.md) — 将 UniUni 置于多机器人协调范式中；产品页侧重预编排执行。
- [Unitree](./unitree.md) — 宇树的软件与硬件生态。
- [Unitree G1](./unitree-g1.md) — UniUni 页面展示的宇树人形机器人之一；具体兼容性仍需核对。
- [UniStore](./unitree-unistore.md) — 动作内容分发平台；UniUni 页面称动作市场与 UniStore 联动。
- [导航、SLAM 与自主系统栈](../overview/navigation-slam-autonomy-stack.md) — 点云地图与定位工作流的上层技术背景。

## 推荐继续阅读

- UniUni 官方产品页：<https://uniuni.unitree.com/>
- UniStore 官方门户：<https://unistore.unitree.com/>
- [人形多机协调](../concepts/humanoid-multi-robot-coordination.md)

## 参考来源

- [UniUni 官方产品页归档](../../sources/sites/unitree-uniuni.md)
- UniUni 官方页面：<https://uniuni.unitree.com/>（核查日期：2026-10-20）
