# MotorFit 官方技术文档归档

- **类型：** site / 官方工程指南
- **项目主页：** https://motorfit.github.io/
- **代码：** https://github.com/motorfit/motorfit
- **相关仓库归档：** [MotorFit](../repos/motorfit.md)
- **核查版本：** main 的 docs 原文；2026-11-17
- **核对范围：** 通读 [快速开始](https://github.com/motorfit/motorfit/blob/main/docs/getting-started/quickstart.md)、[实验设计](https://github.com/motorfit/motorfit/blob/main/docs/getting-started/workflow.md)、[架构](https://github.com/motorfit/motorfit/blob/main/docs/development/architecture.md)、[采集](https://github.com/motorfit/motorfit/blob/main/docs/guide/acquisition.md)、[拟合](https://github.com/motorfit/motorfit/blob/main/docs/guide/identification.md)、[验证](https://github.com/motorfit/motorfit/blob/main/docs/guide/validation.md)、[参数](https://github.com/motorfit/motorfit/blob/main/docs/reference/parameters.md)、[后端](https://github.com/motorfit/motorfit/blob/main/docs/reference/backends.md)和[RS05 案例](https://github.com/motorfit/motorfit/blob/main/docs/case-studies/rs05.md)。未实际运行机电台架和后端。
- **通读：** 已完成
- **映射到：** [MotorFit 实体](../../wiki/entities/motorfit.md)

## 核心原文信息

- 实验先配置电机、夹具质量/惯量及零位，再按负载/增益/轨迹采集独立完整记录；失败采集须检查反馈、时间戳、设备故障与清理状态。
- M1–M6 通过 BAM 拟合；均应先划分 train/validation/final-test；fit --eval 对已选模型评分，不重新拟合。
- replay 依赖 portable dataset manifest 与固定模型；目前跨物理引擎仅 M1 torque-MIT 单摆。MuJoCo armature/frictionloss/damping 只对应部分机械参数，命令延迟和扭矩缩放必须复现控制律。
- 实验中的 111 条 RS05 记录仅有 3 条完整示例随 Git 提供，其余数据和 CAD 存放于独立归档。
- 原文中的 5 mrad 平均轨迹误差门槛是特定回放检查，不是通用执行器验收标准。
