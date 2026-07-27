# Embodied-AI-Guide (Lumina 具身智能社区百科全书)

- **标题**: Embodied-AI-Guide
- **链接**: https://github.com/tianxingchen/Embodied-AI-Guide
- **类型**: repo / encyclopedia
- **作者**: tianxingchen (Lumina Community)
- **社区官网**: https://lumina-embodied.ai/ （博客镜像：https://lumina-embodied.ai/blog/eai-guide）
- **站点归档**: [lumina-embodied-ai.md](../sites/lumina-embodied-ai.md)
- **核心关注点**: 具身智能全栈技术、VLA 模型、仿真到现实管线、RoboTwin 2.0
- **最后核查**: 2026-07-27

## 核心内容摘要

### 1. 技术栈地图 (Algorithm Capability Stack)
该指南将具身智能算法能力拆解为：
- **感知 (Vision & Perception)**: 2D/3D/4D 视觉、视觉提示、Affordance 学习。
- **规划 (Planning)**: 基于 LLM 的任务规划。
- **策略 (Policy)**: VLA (Vision-Language-Action) 模型，强调分层双系统架构。
- **执行 (Action)**: ACT (Action Chunking with Transformers) 等模仿学习策略。

### 2. 关键工具与平台
- **RoboTwin 2.0**: 该指南推荐的手操学习平台，基于 SAPIEN 引擎，提供 50+ 双臂自动数据合成任务。
- **SAPIEN**: 核心仿真引擎，支持基于 PartNet-Mobility 的部件级交互。
- **ALOHA**: 经典的低成本双臂遥操作与数据采集硬件标准。

### 3. 行业趋势 (State of Robot Learning 2025)
- 强调 **Video-as-Simulation** 趋势，视频生成模型可能替代传统物理引擎。
- 强调数据规模化 (Scaling Laws) 在具身智能中的应用。

## 对 Wiki 的映射
- **wiki/entities/lumina-embodied.md** — 社区门户实体（官网 Talks / 教程 / Call）
- **wiki/entities/robotwin.md**
- **wiki/entities/sapien.md**
- **wiki/entities/aloha.md**
- **wiki/methods/vla.md** (Algorithm Capability Stack)
- **wiki/methods/action-chunking.md** (关联 ALOHA 硬件)
