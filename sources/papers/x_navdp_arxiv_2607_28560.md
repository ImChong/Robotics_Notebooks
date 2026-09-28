# X-NavDP：GQRM 跨本体导航扩散策略 RL 后训练（arXiv:2607.28560）

> 来源归档（ingest）

- **英文标题：** X-NavDP: Generalizing Navigation Diffusion Policy to Novel Behavior and Embodiments with Group Q-score Reweighted Matching
- **类型：** paper
- **作者：** Tianyu Yang, Yiming Zeng, Wenzhe Cai, Yuqiang Yang, Jiaqi Peng, Hui Cheng, Jiangmiao Pang, Tai Wang（* 同等贡献；Tai Wang 通讯）
- **机构：** Fudan University；Shanghai AI Laboratory；Sun Yat-sen University；Tsinghua University
- **arXiv：** <https://arxiv.org/abs/2607.28560>
- **PDF：** <https://arxiv.org/pdf/2607.28560>
- **Hugging Face Paper：** <https://huggingface.co/papers/2607.28560>
- **项目页：** <https://yty-sky.github.io/x-navdp-project-page/>
- **代码 / 权重：** <https://github.com/InternRobotics/NavDP>（`baselines/x-navdp`）；资产与 checkpoint：<https://huggingface.co/InternRobotics/X-NavDP>
- **项目页源码（静态站）：** <https://github.com/yty-sky/x-navdp-project-page>
- **开源：** **已开源**（MIT，X-NavDP baseline；依赖 Isaac Sim 5 / Isaac Lab 0.46、acados、Scene-N1 等外部资产）
- **入库日期：** 2026-09-28

## 核心论文摘录

### 1) GQRM：Group Q-score Reweighted Matching

- 扩散导航策略 RL 后训练：自举轨迹扰动（goal / no-goal 混合）+ **组内 Q-score 归一化** 做 reweighted score matching，避免似然梯度不稳定。
- **对 wiki 的映射：** [../../wiki/entities/paper-x-navdp.md](../../wiki/entities/paper-x-navdp.md)

### 2) 跨本体与 Embodiment FiLM

- 在 [NavDP](../../wiki/entities/paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md) 骨干上注入 embodiment 信息（FiLM）；分布式在线 RL 覆盖 Dingo / Unitree Go2 / Unitree G1。
- **对 wiki 的映射：** [../../wiki/entities/paper-x-navdp.md](../../wiki/entities/paper-x-navdp.md)

### 3) 仿真与真机 hard case

- 40 held-out scenes：Overall SR **61.20%→84.28%**，SPL **58.95%→77.19%**；真机 zero-shot hard case 平均 SR **10%→65%**；后训练约 **12 h**。
- **对 wiki 的映射：** [../../wiki/entities/paper-x-navdp.md](../../wiki/entities/paper-x-navdp.md)

## 当前提炼状态

- [x] 项目页 / HF / NavDP baseline 步骤 2.5 核查
- [x] wiki 论文实体页
