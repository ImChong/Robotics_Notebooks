---
type: entity
project_id: submillisecond-reflex-engine
tags: [paper, robotics, ingest-2026-11]
status: draft
updated: 2026-11-14
arxiv: "2610.21913"
code: https://github.com/RoboRef/franka_reflex_engine
related:
  - ./paper-carryon-dual-loop.md
  - ./paper-trace-conversational-humanoid-motion.md
original:
  - ../../sources/papers/submillisecond-reflex-engine_arxiv_2610_21913.md
sources:
  - ../../sources/papers/submillisecond-reflex-engine_arxiv_2610_21913.md
  - ../../sources/repos/submillisecond-reflex-engine.md
summary: "ROS 2 兼容的亚毫秒反射引擎"
---

# 亚毫秒反射：连续人机物理交互的无锁控制切换

## 一句话定义

**以可并存的多控制器与无锁切换机制，在连续控制中快速切换运动、柔顺与安全反射控制行为。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| pHRI | Physical Human-Robot Interaction | 物理人机交互 |
| ROS 2 | Robot Operating System 2 | 机器人中间件 |
| Jerk | Rate of Change of Acceleration | 加速度变化率 |

## 为什么重要

ROS 2 兼容的亚毫秒反射引擎，关注现有控制、数据或机器人学习方法在复杂任务中的失效点。这里单独建论文实体，避免与其他论文共用详情页；当前是待完成全文与源码复核的草稿。

## 核心机制

ROS 2 兼容 multimode controller 将控制行为预备于实时循环中；危险接触、关节极限和速度阈值触发 reflex 状态机，通过无锁指针切换活跃控制器而非停机重启。

## 方法流程（待全文逐节复核）

以下流程仅依据已核对的论文 HTML 方法描述，不代替源代码调用图。

```mermaid
flowchart LR
  sensors[力矩速度与接触监测] --> hazard[反射危险判定]
  task[任务控制器] --> switch[无锁控制器选择]
  hazard --> switch
  switch --> control[实时多模态控制器]
  control --> robot[Franka 执行器]
  robot --> sensors
```

## 源码运行时序图

**待核实，不绘制未经代码阅读确认的时序图。** 官方存在公开仓库，但本次仅核对了 README 和部分发布说明；需继续阅读真实运行入口及关键模块后补充。

## 实验与评测

Franka Panda 的控制模式切换延迟低于 0.1 ms；多阶段任务执行耗时降低 3.5%，峰值 jerk 下降超过三倍。0.1 ms 是控制模式切换耗时，非接触后机械系统卸力的总时延。

## 局限与工程实践

- 结果针对所测 Franka 与控制软件栈；实际安全性能仍取决于传感、驱动、执行器动态与设备安全机制。
- **源码状态：** 公开 ROS 2 / Franka reflex engine，README 说明依赖 multipanda_ros2；实际部署需核对硬件/版本。
- **原文状态：** 本轮已核查 arXiv 部分章节，尚未通读全部正文/附录，保持 draft，不宣称健康度满分。

## 结论

**反射切换可以优化控制软件实时路径，但不能把软件切换时延当作完整机械安全响应时间。**

1. 使用与论文一致的任务、硬件和评测协议复验，不跨实验口径比较数据。
2. 把方法改善的指标与额外的数据、计算或控制成本分开记录。
3. 对未经独立核查的复现入口，暂不声称完成验证。

## 关联页面

- [CarryOn：具备记忆、安全与失败恢复的双循环机器人 Harness](./paper-carryon-dual-loop.md)
- [TRACE：具有参考保持约束的对话式人形动作编辑](./paper-trace-conversational-humanoid-motion.md)

## 参考来源

- [论文归档](../../sources/papers/submillisecond-reflex-engine_arxiv_2610_21913.md)
- [官方仓库归档](../../sources/repos/submillisecond-reflex-engine.md)

## 推荐继续阅读

- [arXiv 原文](https://arxiv.org/abs/2610.21913)
- [公开仓库](https://github.com/RoboRef/franka_reflex_engine)
