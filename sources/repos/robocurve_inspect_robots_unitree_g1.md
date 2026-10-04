# robocurve/inspect-robots-unitree-g1

> 来源归档

- **标题：** Inspect Robots adapters for Unitree G1 humanoids driven by GR00T policy servers
- **类型：** repo / robotics adapter
- **组织：** Robocurve
- **代码：** <https://github.com/robocurve/inspect-robots-unitree-g1>
- **PyPI：** <https://pypi.org/project/inspect-robots-unitree-g1/>
- **主框架：** [robocurve/inspect-robots](https://github.com/robocurve/inspect-robots)
- **License：** MIT（仓库 LICENSE）
- **状态：** 开源适配器；主框架 API 与机器人固件/GR00T seam 均需锁定验证
- **复核日期：** 2026-10-04
- **沉淀到 wiki：** [Inspect Robots](../../wiki/entities/inspect-robots.md)

## 一句话说明

为 Inspect Robots 注册 Unitree G1 的 g1_arms Embodiment 与 gr00t Policy，通过 GR00T PolicyServer 评测站立 G1 的双臂任务；腿、腰和平衡仍由 G1 原 locomotion controller 负责。

## 接口与数据路径

| 组件 | 官方 README 所述职责 |
|------|----------------------|
| g1_arms Embodiment | 读取 G1 双臂状态与头部 D435i 图像；向 rt/arm_sdk 发布臂目标 |
| gr00t Policy | 作为 GR00T PolicyServer 客户端；模型与 checkpoint 由用户另行准备 |
| 动作契约 | 16 维绝对关节位置；README 要求 GR00T checkpoint / server 的 embodiment tag 对齐 |
| 控制路径 | 策略服务生成 chunk → 适配器将臂/手目标接入 G1 arm-sdk → 原全身控制器继续负责支撑与平衡 |
| 配置快照 | 文档默认 control_hz=10、同步发布 stream_hz=50；实际频率须服从 checkpoint 数据帧率 |

适配器通过 inspect-robots-unitree-g1-preflight --dry-run 校验声明层面的空间、语义、相机/状态键和可选场景。预检不连接硬件或 PolicyServer，也无法发现绝对/相对输出约定、夹爪方向、网络可达性和安全间隙问题。

## 部署风险摘录

- 首次 reset 会从观测到的当前臂姿态种子化目标，并逐步 ramp arm-sdk weight；直接跳变 weight 可能令手臂高速抽动。
- 文档提示公开资料未说明 arm-sdk firmware watchdog；进程异常退出时，控制器可能保持最后目标。运行时需操作者在急停遥控器旁监督。
- stock Isaac-GR00T 特定 API seam 会在服务端把相对输出解码为绝对物理量；actions_are_relative 配置不匹配可能造成目标失控，接口检查不能自动识别。
- control_hz 必须符合训练/checkpoint 数据帧率；仅 shape 相同并不证明策略时序正确。
- README 提醒先低速验证 Dex1 夹爪极性，再进行抓取。手型/硬件修订差异需要本机确认。

> 以上属于仓库 README 的接入说明，不构成安全认证；首次运行应按当前版本文档与现场风险控制执行。

## 参考来源

- [Inspect Robots G1 adapter README](https://github.com/robocurve/inspect-robots-unitree-g1)
- [主框架 Concepts](https://docs.inspectrobots.org/guide/concepts/)
