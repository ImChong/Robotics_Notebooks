# menlo_noise_is_all_you_need

> 来源归档（blog）

- **标题：** Noise is all you need to bridge the sim-to-real locomotion gap
- **类型：** blog
- **来源：** Menlo Research（menlo.ai）
- **原始链接：** https://menlo.ai/blog/noise-is-all-you-need
- **原文日期：** 2026-02-12（页面标注）
- **入库日期：** 2026-05-14
- **一句话说明：** 在 MuJoCo 中运行**未改动的生产固件**，用 I2C/CAN 外设仿真与**注入式总线抖动**把线程时序、融合库数值语义与协议解析纳入训练闭环，从工程上把 sim2real 从「纯物理 gap」扩展到**嵌入式与通信层可观测性**问题。

## 为什么值得保留

- **问题陈述清晰**：指出真机失败常来自 CAN 迟到、线程错过周期、IMU 积分漂移等**非摩擦百分之几**类因素，与主流「只补物理模型 + DR」叙事形成对照。
- **方法可命名**：提出 **Processor-in-the-loop（处理器在环）**——控制环经真实固件执行模型闭合，处理器线程、时序与数值约束被视为**动力学的一部分**。
- **可检验的工程细节**：MuJoCo 的 `lsm6dsox` 模型经 **I2C 仿真器**按寄存器语义流式输出；固件用 **xioTechnologies/Fusion** C 库做投影重力等估计；**CAN 总线仿真器**让固件以为在连真硬件；**电机仿真器**在请求–响应路径上注入 **0.4–2 ms 均匀随机延迟**（对照 datasheet 名义 0.4 ms），把「DR  actuator delay」变成**可对固件/策略同时施压的集成测试**。
- **与 Asimov 全栈叙事一致**：文中以 Asimov Legs 与 **12 Series 电机**部署为例，展示零样本 sim2real 行走与推扰恢复，并给出真机 vs 仿真关节轨迹对照思路。

## 核心摘录

### 1) 不要只仿真机器人，要仿真嵌入式环境

- **要点：** 把处理器纳入机器人动力学；固件在 deliberately imperfect 仿真里跑，消费仿真 IMU、在真实时序与通信约束下输出力矩指令。
- **对 wiki 的映射：**
  - [processor-in-the-loop-sim2real](../../wiki/concepts/processor-in-the-loop-sim2real.md)
  - [sim2real](../../wiki/concepts/sim2real.md)
  - [asimov-v1](../../wiki/entities/asimov-v1.md)

### 2) 抖动与延迟域需要被「优化」，而非只在 datasheet 上假设

- **要点：** 名义响应时间在完美仿真里常变成瞬时常量；真实世界波动。随机延迟可暴露线程竞态、CAN 驱动解析错误、策略对抖动不耐受、传感器读数与控制节拍错位导致的融合漂移等。
- **对 wiki 的映射：**
  - [domain-randomization](../../wiki/concepts/domain-randomization.md)
  - [sim2real-gap-reduction](../../wiki/queries/sim2real-gap-reduction.md)

### 3) Sim2Real 无法在黑盒里解决

- **要点：** 闭源固件隐藏主导失效模式；sim2real 被表述为**可观测性**问题——测不到就修不了。与硬件–固件–策略**协同设计**（三层并行演化而非最后硬凑）同一论点。
- **对 wiki 的映射：**
  - [open-source-humanoid-hardware](../../wiki/entities/open-source-humanoid-hardware.md)
  - [sim2real](../../wiki/concepts/sim2real.md)

## 原文简摘

- 行业常把腿式 sim2real 框成物理 gap（接触、摩擦、Sim2Sim 对齐）；作者认为大量现场失效来自**嵌入式与通信噪声**。
- 管线：**MuJoCo 传感器模型 → I2C 仿真 → 真 Fusion 栈（嵌入式浮点语义）→ CAN 仿真（二进制级协议）→ 电机路径抖动注入**；目标是在 PR/CI 阶段捕获通常只在现场出现的失效模式。
- 展示 Asimov 上单一策略的零样本行走与推恢复，并建议用关节轨迹图做跨域对齐检查。

## 当前提炼状态

- [x] 文章基础摘要填写
- [x] wiki 概念页 `processor-in-the-loop-sim2real.md` 已创建并互链
- [x] `sim2real.md` / `asimov-v1.md` 交叉引用已补充
