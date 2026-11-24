# Intrinsic Core：工业机器人本地运行时与实时控制框架（Industrial Robotics Runtime & Real-Time Control） — repos 来源归档

- **类型：** repo
- **标题：** Intrinsic Core：工业机器人本地运行时与实时控制框架（Industrial Robotics Runtime & Real-Time Control）
- **机构：** Intrinsic
- **原文：** https://github.com/intrinsic-ai/intrinsic-core
- **项目主页：** https://developer.intrinsic.ai/
- **代码：** https://github.com/intrinsic-ai/intrinsic-core
- **开放状态：** Apache-2.0；官方 README 明确为本地开放运行时及 SDK。外部依赖及设备驱动支持需逐项确认。
- **入库日期：** 2026-11-24
- **核对范围：** 官方 README 全文、Getting Started 部署教程、教程索引、AdderSkill::Execute 示例源码；不是对全部 ICON 实现的逐行审计。
- **通读：** 已完成
- **对应知识页：** [intrinsic-core](../../wiki/entities/intrinsic-core.md)

## 资料要点

- intrinsic_runtime 负责进程生命周期和应用状态；intrinsic_sdk / intrinsic_apis 为技能与资源定义接口；intrinsic_perception 与 intrinsic_inference 提供传感、6-DoF 位姿和边缘推理；intrinsic_motion_planning 负责约束路径与轨迹生成；ICON 执行实时控制并经 HAL 连接驱动器。各能力可组合，不表示每个技能都必须经过所有模块。
- 官方 README 给出开源机床上下料 OMTS 参考解决方案，并声明可通过统一 HAL 更换机械臂与夹爪。未见本次读取资料给出统一可复现的周期抖动、负载下延迟或跨硬件性能数字，不编造指标。
- 硬件无关接口不代表任意设备无需适配；实际现场仍要验证实时性、驱动兼容、急停和安全认证。开发者社区页面未能提取正文，已核对部署教程，ICON 实现未逐行审计。

## 原文通读补充（2026-11-24）

### 本地部署与模块边界

- README 列出 runtime、ICON、motion planning、perception、inference、SDK、API、hardware 与 kinematics 九类模块。ICON 结合轨迹插值、运动学及确定性控制循环，HAL 负责设备适配；感知模块支持 FoundationPose 6-DoF 估计，规划模块负责约束、限位与轨迹衔接。
- 官方 Getting Started 以 k3s 本地集群部署参考 OMTS 为主线：安装工具及依赖、部署运行时、用 Bazel 构建与部署参考方案、通过可视化/调试接口检查。README 标注 Ubuntu 24.04/26.04（亦支持 22.04）和 ROS 2 Lyrical Luth；实际兼容性须按发布版本确认。

### 工程核验与限制

- OMTS 是机床上下料参考方案，不能由参考方案推导所有设备均可免驱动适配；需要按机器人型号验证 HAL、现场总线、控制周期、故障及急停。SDK AdderSkill 例程只验证技能接口调用关系，不是实时运动循环。未进行实机或部署验证。

## 公开资源

- [官方项目页](https://developer.intrinsic.ai/)
- [官方源码](https://github.com/intrinsic-ai/intrinsic-core)
- [OMTS 机床上下料参考方案](https://github.com/intrinsic-ai/intrinsic-omts)
- [Getting Started](https://github.com/intrinsic-ai/intrinsic-core/blob/main/developer_resources/learn/tutorials/getting_started.md)

## 对 wiki 的映射

- [intrinsic-core 独立实体](../../wiki/entities/intrinsic-core.md)
- [关联归档](../sites/intrinsic-core.md)
