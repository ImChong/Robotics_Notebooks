# Jumper — KingKongRobotics

> 来源归档

- **标题：** Jumper — an open-source crab robot
- **类型：** repo
- **来源：** KingKongRobotics
- **链接：** https://github.com/KingKongRobotics/jumper
- **许可证：** 仓库自有材料 Apache-2.0；第三方组件和素材按各自许可证，见仓库 NOTICE。
- **入库日期：** 2026-10-04
- **一句话说明：** 六足仿蟹机器人 Jumper 的设计资料、MuJoCo/mjlab 强化学习训练、仿真、策略导出和控制器打包工程。
- **沉淀到 wiki：** 是 → [Jumper 六足机器人与训练部署栈](../../wiki/entities/jumper-crab-robot.md)

---

## 开源状态

截至 2026-10-04，训练、仿真、评估、导出和部署打包代码已公开；Apache-2.0 仅覆盖仓库自有材料，依赖及第三方素材需查 NOTICE 和上游许可。开源不等于全链路真机验证：项目指南将 RKNN 板端推理和 1 kHz 电机总线闭环列为尚未验证。

## 项目定位与结构

Jumper 是一台仿蟹式六足机器人。主仓包含 MuJoCo 资产、mjlab 任务和奖励、训练与评估脚本、策略导出/打包工具及 Rust 控制器。训练栈使用 mjlab 与 RSL-RL PPO；GPU 仿真主路径是 MuJoCo Warp，同时保留原生 MuJoCo CPU 路径。

- 训练入口：scripts/train.py、scripts/play.py
- 策略导出：scripts/export.py，输出 ONNX 与 layout.json
- 部署打包：scripts/deploy.py，组合各模式策略、FSM 与控制映射
- 硬件规格：docs/HARDWARE.zh.md
- 任务/仿真/部署边界：docs/PROJECT_GUIDE.zh.md、docs/WORKFLOWS.md、docs/DESIGN.md
- 外观和场景工具是独立仓库 [jumper-design](https://github.com/KingKongRobotics/jumper-design)。它不属于训练仓库内的隐式依赖；map、skin 的导入、训练和回放边界需要按文档分别处理。

## 关键实现线索

| 层 | 仓库实现 | 阅读时要核对 |
|---|---|---|
| 任务 | tripodal 速度跟踪、跳跃、舞蹈的独立配置 | 奖励和参考信号不同，不是同一个策略换名称 |
| 训练 | mjlab manager 环境 + RSL-RL PPO | 观测、奖励、随机化和终止逻辑在任务/环境层 |
| 仿真 | MuJoCo Warp GPU 与 native MuJoCo CPU 后端 | 共享上层接口不意味着接触数值完全相同 |
| 导出 | PyTorch actor → ONNX + layout.json | 检查观测是否可测、关节是否按名称对齐 |
| 部署 | ONNX → RKNN 转换及统一 bundle；Rust FSM | 打包通过不能替代板端 NPU、总线时序和实机验证 |

## 复现入口

`bash
python scripts/train.py --task jumper.tripod --num_envs 4096
python scripts/play.py --task jumper.tripod --checkpoint <checkpoint>
python scripts/export.py --task jumper.tripod --checkpoint <checkpoint>
`

命令参数以当前配置和脚本 --help 为准。教程还覆盖 jumper.jump 与 jumper.dance；跳跃/舞蹈策略依赖随策略契约提供的参考轨迹，不能只复制权重文件便视为完整部署包。

## 官方文档

- [项目指南](https://github.com/KingKongRobotics/jumper/blob/main/docs/PROJECT_GUIDE.zh.md)
- [训练教程](https://github.com/KingKongRobotics/jumper/blob/main/docs/TUTORIAL.zh.md)
- [仿真设计](https://github.com/KingKongRobotics/jumper/blob/main/docs/DESIGN.md)
- [部署说明](https://github.com/KingKongRobotics/jumper/blob/main/deploy/README.zh.md)
- [硬件规格](https://github.com/KingKongRobotics/jumper/blob/main/docs/HARDWARE.zh.md)
- [LICENSE](https://github.com/KingKongRobotics/jumper/blob/main/LICENSE) 与 [NOTICE](https://github.com/KingKongRobotics/jumper/blob/main/NOTICE)
