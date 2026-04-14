# simulation_tools

> 来源归档（ingest）

- **标题：** 机器人仿真工具核心论文
- **类型：** paper
- **来源：** IROS / NeurIPS / arXiv
- **入库日期：** 2026-04-14
- **最后更新：** 2026-04-14
- **一句话说明：** 覆盖 MuJoCo、Isaac Gym/Lab、Genesis 等主流机器人仿真框架的原始论文

## 核心论文摘录

### 1) MuJoCo: A Physics Engine for Model-Based Control（Todorov et al., IROS 2012）
- **链接：** <https://ieeexplore.ieee.org/document/6386109>
- **核心贡献：** 提出 soft-contact 模型 + 隐式积分器；接触力通过优化求解；支持肌腱、铰链、自由关节等复杂拓扑；比 ODE/Bullet 在机器人学习场景下更稳定快速；2022 年开源
- **对 wiki 的映射：**
  - [mujoco](../../wiki/entities/mujoco.md)

### 2) Isaac Gym: High Performance GPU-Based Physics Simulation（Makoviychuk et al., NeurIPS 2021）
- **链接：** <https://arxiv.org/abs/2108.10470>
- **核心贡献：** 在单 GPU 上并行运行 4096 个仿真环境；CPU-GPU 数据拷贝开销降至几乎为零；legged_gym + AnymalC 示例展示 10 分钟 RL 训练；推动了"单机训练真机可用策略"范式
- **对 wiki 的映射：**
  - [isaac-gym-isaac-lab](../../wiki/entities/isaac-gym-isaac-lab.md)
  - [legged-gym](../../wiki/entities/legged-gym.md)
  - [sim2real](../../wiki/concepts/sim2real.md)

### 3) Isaac Lab: A Unified Modular Platform for Robot Learning（Mittal et al., RAL 2023）
- **链接：** <https://arxiv.org/abs/2301.04195>
- **核心贡献：** 在 Isaac Sim（Omniverse）基础上统一 RL/IL/迁移学习管线；模块化 Task/Environment/Manager 设计；兼容 OpenAI Gym / RL Games / rsl_rl；取代 Isaac Gym 成为 NVIDIA 官方训练框架
- **对 wiki 的映射：**
  - [isaac-gym-isaac-lab](../../wiki/entities/isaac-gym-isaac-lab.md)

### 4) Genesis: A Generative and Universal Physics Engine（Genesis Team, 2024）
- **链接：** <https://arxiv.org/abs/2412.12919>
- **核心贡献：** 统一多种物理求解器（刚体/弹性体/流体/SPH/FEM）在单框架内；支持可微分仿真；比 Isaac Gym 快 10-80x（CPU 并行）；开源于 2024 年底，预期成为下一代仿真标准
- **对 wiki 的映射：**
  - [isaac-gym-isaac-lab](../../wiki/entities/isaac-gym-isaac-lab.md)
  - [mujoco](../../wiki/entities/mujoco.md)
  - [domain-randomization](../../wiki/concepts/domain-randomization.md)
  - [sim2real](../../wiki/concepts/sim2real.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [ ] 关联 wiki 页面的参考来源段落已添加 ingest 链接
