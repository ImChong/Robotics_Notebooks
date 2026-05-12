# Source: MimicKit (xbpeng/MimicKit)

- **Title**: MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control
- **URL**: https://github.com/xbpeng/MimicKit
- **Project page**: https://motion.stanford.edu/research/mimickit
- **Paper**: https://arxiv.org/abs/2510.13794
- **Author**: Xue Bin Peng
- **Year**: 2025
- **Type**: Codebase / Research Framework

## 核心内容
- **定位**：轻量级、模块化的强化学习框架，专为物理机器人/角色的运动模仿与控制设计。
- **架构**：
    - 支持 Isaac Gym, Isaac Lab, Newton 等多种模拟引擎。
    - 使用 YAML 配置驱动，支持 3D 指数映射（Exponential Maps）表示旋转。
    - 提供从 AMASS (SMPL) 等数据集的重定向（Retargeting）工具。
- **集成算法**：DeepMimic, AMP, AWR, ASE, LCP, ADD, SMP。
