# Xbotics-Embodied-Guide（Xbotics 具身智能学习指南 / Roadmap to Embodied AI）

- **类型：** repo / 具身智能学习路线 / 社区文档指南（非独立训练框架）
- **官方仓库：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide
- **作者与维护组织：** Xbotics-Embodied-AI-club（原入口 mumu-jushen/Xbotics-Embodied-Guide 已迁移至组织）
- **核对版本：** [main@19b9e947b4d4](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide/tree/19b9e947b4d4e355655a94691ab51c6ce1cbe6f7)；核对日 2026-11-22
- **核对范围：** 通读仓库 README、`docs/2-roadmaps/README.md`（十方向主路线）、`docs/1-embodied-overview/README.md`、`docs/3-foundations/README.md`、`docs/4-classical/README.md`、`docs/5-sota/README.md`、`docs/6-simulation/README.md`、`docs/7-real-robots/README.md`、`docs/10-contributing/README.md` 与根目录 `LICENSE`；核对 `docs/8-people/README.md` 人物页结构及 `docs/9-landscape/README.md` 公司分类，未逐条外部验证人物／公司名录，也未审计仓内 `files/` 的全部延伸资料。
- **通读：** 已完成
- **项目形态与开放状态：** 文档和案例命令公开可读；上游 README 明确“以文档与路线为主，无需运行脚本即可开始学习”。`docs/7-real-robots/README.md` 提供外部 LeRobot 的安装与训练命令，不能据此认定本仓有独立可运行的训练或推理实现。
- **许可证核对：** README 标注 CC BY 4.0，`docs/10-contributing/README.md` 声称代码 MIT / 文档 CC BY 4.0；但根目录 `LICENSE` 文本实际为 CC BY-NC-SA 4.0。三处不一致，转载或商业改编前需向维护者确认，不能单凭 README 声明推定许可范围。

## 原文内容与定位

作者主张“具身综述 → 学习路线 → 仿真学习 → 开源实物 → 人物与公司图谱”；不是百科全书或端到端机器人软件，而是让读者从技术地图选择方向，完成开源实践并积累可展示成果。

| 模块 | 原文入口 | 核对到的具体内容 |
| --- | --- | --- |
| 领域综述 | [1. Embodied Overview](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide/blob/19b9e947b4d4e355655a94691ab51c6ce1cbe6f7/docs/1-embodied-overview/README.md) | 术语、操作、世界模型、运动控制与导航 |
| 学习路线 | [2. Roadmaps](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide/blob/19b9e947b4d4e355655a94691ab51c6ce1cbe6f7/docs/2-roadmaps/README.md) | RL、IL、3D 视觉、规划控制、定位导航、力控触觉、VLA、Sim2Real、世界模型、数据飞轮与遥操作等十方向 |
| 理论与前沿 | [3–5 章](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide/tree/19b9e947b4d4e355655a94691ab51c6ce1cbe6f7/docs) | 机器人学基础、BC/DAgger/GAIL、Transformer/扩散模型、VLA 等参考资料 |
| 仿真学习 | [6. Simulation](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide/blob/19b9e947b4d4e355655a94691ab51c6ce1cbe6f7/docs/6-simulation/README.md) | Isaac Lab、MuJoCo、PyBullet、Genesis、Gazebo 的入门资料链接 |
| 真机工具 | [7. Real Robots](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide/blob/19b9e947b4d4e355655a94691ab51c6ce1cbe6f7/docs/7-real-robots/README.md) | 以 LeRobot 为核心的安装、数据集、训练／评估命令示例；SO-ARM、ALOHA 等硬件索引 |
| 行业与贡献 | [8–10 章](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide/tree/19b9e947b4d4e355655a94691ab51c6ce1cbe6f7/docs) | 人物、公司、AMA、贡献规范；README 另列求职 Roadmap |

### 学习路线实质

- **路线入口：** 先按目标选 10 个方向之一，再按前置条件 → 理论/工具 → 最小目标 → 进阶目标完成实践。
- **上游模板：** 提议 4–8 周计划、里程碑、作业和验收，但并非每条路线都已具体写满分周计划；不能将模板当成已经完成的十套课程。
- **可交付产物：** 由最小 Demo 进阶为改造实验与指标报告；README 求职路线建议把复现、README、指标和演示视频整理为作品集。
- **项目分工：** 与 [Xbotics-Embodied-AI-Handbook](./xbotics-embodied-ai-handbook.md) 互补：本 Guide 负责路线导航，Handbook 负责 21 讲动手教程。

### 工程使用边界

1. 选 VLA/模仿学习，先补 PyTorch 与策略理论，再用 LeRobot / ACT / Diffusion Policy 生态做可复现练习；具体 CLI 以当前 LeRobot 官方文档为准。
2. 选强化学习或 Sim2Real，先在模拟器完成训练和跨仿真验证，再建立真实系统的参数、控制频率、安全检查与失败回流；这是一种**学习规划建议**，并非上游仓库发布的完整可执行 SOP。
3. 人物与公司图谱适合探索方向，官网、组织归属与招聘状态需要二次核查。
4. 仓内资料部分沿用旧版项目接口或年份描述，不能未经验证直接作为 2026 年依赖版本、性能或实验指标的事实依据。

## 对 Wiki 的映射

- **主实体：** [Xbotics-Embodied-Guide](../../wiki/entities/xbotics-embodied-guide.md)（本次补齐唯一独立详情节点）。
- **配套实践课：** [Xbotics-Embodied-AI-Handbook](../../wiki/entities/xbotics-embodied-ai-handbook.md)。
- **关联概念：** [LeRobot](../../wiki/entities/lerobot.md)、[Genesis](../../wiki/entities/genesis-sim.md)、[数据飞轮](../../wiki/concepts/data-flywheel.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[VLA](../../wiki/methods/vla.md)。
- **纵深路线：** [Sim2Real](../../roadmap/depth-sim2real.md) Stage 4 作为外部学习导航和实践清单参考。

## 一手来源

- [项目 README](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide/blob/19b9e947b4d4e355655a94691ab51c6ce1cbe6f7/README.md)
- [十方向 Roadmaps](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide/blob/19b9e947b4d4e355655a94691ab51c6ce1cbe6f7/docs/2-roadmaps/README.md)
- [仿真学习](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide/blob/19b9e947b4d4e355655a94691ab51c6ce1cbe6f7/docs/6-simulation/README.md)
- [开源实物](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide/blob/19b9e947b4d4e355655a94691ab51c6ce1cbe6f7/docs/7-real-robots/README.md)
- [贡献规则](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide/blob/19b9e947b4d4e355655a94691ab51c6ce1cbe6f7/docs/10-contributing/README.md)
- [根目录许可证](https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide/blob/19b9e947b4d4e355655a94691ab51c6ce1cbe6f7/LICENSE)
