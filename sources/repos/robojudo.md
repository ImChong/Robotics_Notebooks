# Source: RoboJuDo（HansZ8/RoboJuDo）

- **Title**: RoboJuDo: A Plug-and-Play Deploy Framework for Robots
- **URL**: https://github.com/HansZ8/RoboJuDo
- **Type**: Humanoid Robot Deployment Framework
- **Relationship**: FRoM-W1 的 sim-to-real 部署模块；独立代码仓库
- **Related project**: [FRoM-W1](https://github.com/OpenMOSS/FRoM-W1)
- **Related paper**: [FRoM-W1: Towards General Humanoid Whole-Body Control with Language Instructions](https://arxiv.org/abs/2601.12799)
- **License**: 仓库 LICENSE 文件正文标注 CC BY 4.0；README 徽章链接的许可地址与文字标签不一致，使用者应以仓库当前 LICENSE 与作者说明核对
- **入库日期**: 2026-10-17

## 核心内容（官方 README 归纳）

- **定位**：面向人形机器人的轻量策略部署框架，目标是在仿真与真机上以统一方式运行不同全身控制与 locomotion 策略。
- **模块边界**：把 Controller（键盘、手柄或动作序列输入）、Environment（MuJoCo 或真机接口）和 Policy（控制策略）拆开，通过配置组合成运行 pipeline。
- **多机器人 / 多环境**：README 列出 Unitree G1、H1 与 FFTAI GR-1T1 等平台，并提供仿真和真机部署路径；具体的策略—机器人—环境组合见仓库支持矩阵。
- **策略切换**：提供多策略 pipeline 与插值过渡示例，可在 locomotion 和 motion-mimic 策略之间切换。
- **策略接入**：README 展示 Human2Humanoid、BeyondMimic、ASAP、TWIST、KungfuBot / KungfuBot2，以及 2026-03 增加的 ProtoMotions G1 tracker 集成。
- **部署依赖**：真实机器人运行需要对应机器人 SDK 与网络接口配置；仓库提供 Unitree SDK / UnitreeCpp 设置路径。README 强调真机运动风险，要求先验证急停并采取严格安全措施。
- **发布边界**：README roadmap 将 HugWBC 与 GMT 的代码列为待发布；不要把支持矩阵中提及或 roadmap 规划中的策略一概视为已公开可运行。

## FRoM-W1 流程中的位置

FRoM-W1 的官方快速开始把 RoboJuDo 放在动作生成与重定向、策略训练之后：它接收机器人策略/轨迹配置，在仿真或实体机器人环境中执行跟踪。RoboJuDo 是可单独复用的部署仓库，但在此处按其官方关系归入 FRoM-W1 项目实体。

## 参考来源（原始）

- [RoboJuDo 官方仓库 README](https://github.com/HansZ8/RoboJuDo)
- [RoboJuDo LICENSE](https://github.com/HansZ8/RoboJuDo/blob/release/LICENSE)
- [FRoM-W1 官方仓库](https://github.com/OpenMOSS/FRoM-W1)
- [FRoM-W1 快速开始](https://github.com/OpenMOSS/FRoM-W1/blob/main/QUICKSTART.md)
- [FRoM-W1 论文](https://arxiv.org/abs/2601.12799)
