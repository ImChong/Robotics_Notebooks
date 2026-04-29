# Xbotics-Embodied-Guide (Xbotics 社区具身智能学习指南)

- **标题**: Xbotics-Embodied-Guide
- **链接**: https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide
- **类型**: repo / engineering-guide / roadmap
- **作者**: Xbotics-Embodied-AI-club
- **核心关注点**: 工程落地、任务驱动学习、LeRobot、Genesis、数据飞轮、Sim2Real SOP

## 核心内容摘要

### 1. 定位：从“查阅”到“实战”
相比于百科全书式的指南（如 tianxingchen/Embodied-AI-Guide），Xbotics 侧重于**可复现的工程路径**。它将具身智能拆解为 10 条为期 4-8 周的专项路线图，每条路径都有明确的里程碑和验收标准。

### 2. 关键工具链
- **Hugging Face LeRobot**: 强调该框架在预训练模型调用和数据采集中的核心作用，是连接算法与开源实物的纽带。
- **Genesis (仿真器)**: 推荐作为新一代具身智能仿真平台，与 Isaac Lab 并列。
- **SOP 模式**: 强调标准作业程序（Standard Operating Procedure），旨在解决开源项目“看时容易做时难”的痛点。

### 3. 技术关键词
- **数据飞轮 (Data Flywheel)**: 强调数据采集、清洗、训练到真机闭环的自动化与规模化。
- **VLA 模型 SFT**: 关注大模型在机器人任务中的微调技术。
- **Sim2Real SOP**: 系统化的从仿真到现实的迁移步骤，涵盖参数辨识与域随机化。

### 4. 产业连接
- 提供了详细的公司图谱和人物访谈，将学术前沿（感知-决策-控制）与产业供应链（电机、减速器、传感器、整机）挂钩。

## 对 Wiki 的映射
- **wiki/entities/lerobot.md** (新建)
- **wiki/entities/genesis-sim.md** (新建)
- **wiki/concepts/data-flywheel.md** (新建)
- **wiki/concepts/sim2real.md** (补充 SOP 与 Xbotics 实践)
- **wiki/methods/vla.md** (补充 SFT 与实践路径)
