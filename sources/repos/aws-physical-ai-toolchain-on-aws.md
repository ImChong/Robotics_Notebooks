# The Physical AI Toolchain on AWS

> 来源归档（ingest · AWS 官方样例仓库）

- **类型：** repository / reference architecture
- **仓库：** <https://github.com/aws-samples/sample-the-physical-ai-toolchain-on-aws>
- **许可证：** Apache-2.0（仓库 README）
- **核查日期：** 2026-10-10
- **一句话说明：** AWS 样例集合用参考架构、基础设施即代码和部署自动化，串联 Physical AI 数据生成、模型训练、软件在环仿真及边缘部署。
- **沉淀到 wiki：** [AWS Physical AI Toolchain](../../wiki/entities/aws-physical-ai-toolchain.md)

## 仓库声明的整体架构

README 将开发闭环归纳为合成数据生成、模型训练、SIL 仿真、Sim-to-Real / HIL 四个支柱，并让 NVIDIA OSMO 等工具负责跨异构算力调度。参考实现覆盖 SageMaker、AWS Batch、EC2、EKS、S3、ECR、FSx for Lustre、Greengrass 与 Jetson 等组件。

仓库列出的可用模块包括 Foundation、Cosmos、Isaac Lab、Isaac GR00T、DreamZero、Isaac Sim、OSMO、Strands Agents 和 Isaac Lab Arena Evaluation；Edge Deployment 在 README 中仍标注为 planned。模块的状态和可部署参数可能随仓库更新而变化，应以目标目录的 README 为准。

## 示例任务与使用边界

仓库提供 UR3 + Robotiq 2F-85 的 pick-and-place 示例及 27 条真实遥操作 episode，用来串起数据转 LeRobot、GR00T 模仿学习、Cosmos 数据生成、Isaac Lab RL 和 TensorRT/Greengrass 部署。它是云端基础设施与工作流的参考集合，不是机器人控制器或可直接用于任意硬件的成品策略；实际部署需要检查账号配额、地区资源、IAM/网络、安全设置及各模型和容器的许可。

## 一手入口

- [README](https://github.com/aws-samples/sample-the-physical-ai-toolchain-on-aws/blob/main/README.md)
- [Apache-2.0 LICENSE](https://github.com/aws-samples/sample-the-physical-ai-toolchain-on-aws/blob/main/LICENSE)
