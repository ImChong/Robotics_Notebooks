# Source: ProtoMotions (NVlabs/ProtoMotions)

- **Title**: ProtoMotions3: An Open-source Framework for Humanoid Simulation and Control
- **URL**: https://github.com/NVlabs/ProtoMotions
- **Authors**: Chen Tessler, Yifeng Jiang, Xue Bin Peng, et al.
- **Year**: 2025
- **Type**: Research Framework / Simulation Engine

## 核心内容
- **定位**：由 NVIDIA 开发的高性能、GPU 加速的人形机器人仿真与控制框架。
- **核心功能**：
    - **大规模学习**：利用 GPU 加速，支持在数万个并行环境中使用 AMASS 等海量动捕数据进行训练。
    - **Sim2Real**：提供完整的部署管线，支持在 Unitree G1 等真机上运行。
    - **一键重定向**：集成基于 PyRoki 的优化重定向，快速处理 SMPL 到机器人的映射。
    - **多引擎支持**：兼容 Newton, MuJoCo, IsaacGym 和 Genesis。
- **与 MimicKit 的关系**：MimicKit 的姊妹项目。MimicKit 更侧重于运动模仿学习的算法实现，而 ProtoMotions 是一个更全面的仿真和大规模控制框架。
