# Intrinsic Core：工业机器人本地运行时与实时控制框架（Industrial Robotics Runtime & Real-Time Control） — repos 来源归档

- **类型：** repo
- **标题：** Intrinsic Core：工业机器人本地运行时与实时控制框架（Industrial Robotics Runtime & Real-Time Control）
- **机构：** Intrinsic
- **原文：** https://github.com/intrinsic-ai/intrinsic-core
- **项目主页：** https://developer.intrinsic.ai/
- **代码：** https://github.com/intrinsic-ai/intrinsic-core
- **开放状态：** Apache-2.0；官方 README 明确为本地开放运行时及 SDK。外部依赖及设备驱动支持需逐项确认。
- **入库日期：** 2026-11-24
- **核对范围：** 已读取 README 和所列源码入口；`intrinsic/skills/examples/adder_skill.cc::AdderSkill::Execute` 是已核对的 SDK 技能调用示例，不是 ICON 实时控制循环；部署入口及构建教程待通读。 尚未完整复核全部必要技术资料。
- **通读：** 未完成
- **对应知识页：** [intrinsic-core](../../wiki/entities/intrinsic-core.md)

## 资料要点

- intrinsic_runtime 负责进程生命周期和应用状态；intrinsic_sdk / intrinsic_apis 为技能与资源定义接口；intrinsic_perception 与 intrinsic_inference 提供传感、6-DoF 位姿和边缘推理；intrinsic_motion_planning 负责约束路径与轨迹生成；ICON 执行实时控制并经 HAL 连接驱动器。各能力可组合，不表示每个技能都必须经过所有模块。
- 官方 README 给出开源机床上下料 OMTS 参考解决方案，并声明可通过统一 HAL 更换机械臂与夹爪。未见本次读取资料给出统一可复现的周期抖动、负载下延迟或跨硬件性能数字，不编造指标。
- 硬件无关接口不代表任意设备无需适配；实际现场仍要验证实时性、驱动兼容、急停和安全认证。开发者社区页面未能提取正文，完整部署教程及 ICON 控制实现仍待复核。

## 公开资源

- [官方项目页](https://developer.intrinsic.ai/)
- [官方源码](https://github.com/intrinsic-ai/intrinsic-core)
- [OMTS 机床上下料参考方案](https://github.com/intrinsic-ai/intrinsic-omts)
- [Getting Started](https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/getting_started.md)

## 对 wiki 的映射

- [intrinsic-core 独立实体](../../wiki/entities/intrinsic-core.md)
- [关联归档](../sites/intrinsic-core.md)
