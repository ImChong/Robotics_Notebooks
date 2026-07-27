# UniSim: Learning Interactive Real-World Simulators（arXiv:2310.06114）

> 来源归档（ingest）

- **标题：** Learning Interactive Real-World Simulators（UniSim）
- **类型：** paper / UniSim / generative simulator / video world model / interactive simulation / real-robot transfer
- **arXiv：** <https://arxiv.org/abs/2310.06114>（PDF：<https://arxiv.org/pdf/2310.06114.pdf>）
- **项目页：** <https://universal-simulator.github.io>（重定向至 <https://universal-simulator.github.io/unisim/>）
- **作者：** Sherry Yang（Mengjiao Yang）、Yilun Du、Kamyar Ghasemipour、Jonathan Tompson、Leslie Kaelbling、Dale Schuurmans、Pieter Abbeel
- **机构：** 加州大学伯克利分校（UC Berkeley）、谷歌 DeepMind（Google DeepMind）、麻省理工（MIT）
- **入库日期：** 2026-07-27
- **一句话说明：** **UniSim** 学习可与真实世界交互的生成式模拟器：用动作条件视频生成做长时程交互式 rollout，支撑在仿真中训 RL / 规划策略并 **零样本** 迁到真机；强调「画面可检查」的视频输出族，而非仅低维 latent。

## 开源状态（项目页核查，2026-07-27）

- **未开源 / 仅项目页演示：** 截至入库日，[unisim 项目页](https://universal-simulator.github.io/unisim/) 提供论文叙事、长时程仿真、RL 仿真→真机、长时程规划演示与相关工作链接，**未挂官方训练/推理 GitHub 或权重**。wiki 与工程实践按 **不可复现官方代码** 处理；勿与后续同名社区项目混淆。

## 摘要级要点

- **定位：** 交互式真实世界模拟器 —— 观察 + 动作 → 下一观察（视频），可闭环 rollout。
- **用途：** 纯仿真训 RL 再真机零样本；长指令串接规划；用生成视频训目标条件 VLM 策略。
- **阅读轴（策展）：** 属「未来图像/视频」输出族；风险是 **画面连续 ≠ 动力学正确**（见物理保真度博客）。
- **关系：** 与 UniPi（文本引导视频策略）、Video Adapter 等决策/视频生成工作同谱；对照 latent 规划线（World Models / PlaNet / Dreamer / TD-MPC2）。

## 核心论文摘录（MVP）

### 1) 交互式视频模拟器

- **链接：** 项目页 Long-Horizon Simulations；论文方法总览
- **摘录要点：** 价值在长 episode 仿真，以支持搜索、规划、最优控制或 RL。
- **对 wiki 的映射：**
  - [UniSim 实体页](../../wiki/entities/paper-unisim.md)
  - [Generative World Models](../../wiki/methods/generative-world-models.md)
  - [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)

### 2) 仿真训 RL → 真机零样本

- **链接：** Reinforcement Learning with UniSim
- **摘录要点：** 策略可完全在 UniSim 中训练后部署真机；降低真机试错成本。
- **对 wiki 的映射：**
  - [world-models-route-03-virtual-sandbox](../../wiki/overview/world-models-route-03-virtual-sandbox.md)
  - [world-model-physics-fidelity-outputs](../../wiki/overview/world-model-physics-fidelity-outputs.md)

### 3) 长时程规划与 VLM 策略数据

- **链接：** Long-Horizon Planning with UniSim
- **摘录要点：** 串接长指令、反复 rollout 生成视频，再训目标条件 VLM；演示零样本真机迁移。
- **对 wiki 的映射：**
  - [Model-Based RL](../../wiki/methods/model-based-rl.md)

## BibTeX

```bibtex
@article{yang2023unisim,
  title   = {Learning Interactive Real-World Simulators},
  author  = {Yang, Mengjiao and Du, Yilun and Ghasemipour, Kamyar and
             Tompson, Jonathan and Kaelbling, Leslie and Schuurmans, Dale and
             Abbeel, Pieter},
  journal = {arXiv preprint arXiv:2310.06114},
  year    = {2023}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-unisim.md`](../../wiki/entities/paper-unisim.md)
- 项目页：[`sources/sites/universal-simulator-github-io.md`](../sites/universal-simulator-github-io.md)
- 博客策展：[`sources/blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md`](../blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md)
