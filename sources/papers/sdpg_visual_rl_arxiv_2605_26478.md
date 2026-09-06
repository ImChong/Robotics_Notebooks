# sdpg_visual_rl_arxiv_2605_26478

> 来源归档（ingest）

- **标题：** Efficient On-policy Visual-RL via Stochastic Decoupled Policy Gradient
- **类型：** paper
- **作者：** Haoxiang You, Yilang Liu, Davis Zong, Qian Wang, Teeratham Vitchutripop, Qi Wang, Daniel Rakita, Ian Abraham（Yale / SJTU / Sydney）
- **arXiv：** <https://arxiv.org/abs/2605.26478>
- **项目页：** <https://haoxiangyou.github.io/sdpg-website/>
- **代码：** <https://github.com/HaoxiangYou/SDPG>（**已开源**）
- **入库日期：** 2026-09-06
- **一句话说明：** 随机解耦策略梯度（Stochastic Decoupled Policy Gradient）：用随机扰动 rollout 估计轨迹梯度，混合 batch-rendered 与 physics-only 并行环境，在单卡 RTX 4080 上数小时内端到端训练视觉 MuJoCo 与 egocentric 机器人任务，并展示 Unitree Go2 零样本 sim2real。

## 核心论文摘录（MVP）

### 1) 问题与动机（Abstract / §1）

- **链接：** <https://arxiv.org/abs/2605.26478>
- **核心贡献：** 视觉 RL 比状态 RL 更耗算力与显存：DrQ-v2 / DreamerV3 需大量序贯更新；PPO 类 on-policy 方法需数千并行环境 batch 渲染（如 4096 env × RGB）易 OOM。蒸馏路线（teacher 状态 → student 视觉）快但受信息不对称与分布偏移限制。一阶可微仿真（decoupled PG）高效但梯度不稳、需软接触、仿真器支持差。
- **对 wiki 的映射：**
  - [SDPG 视觉 RL 论文实体](../../wiki/entities/paper-sdpg-visual-rl-stochastic-decoupled.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)

### 2) 随机解耦策略梯度（§3）

- **链接：** <https://haoxiangyou.github.io/sdpg-website/>
- **核心贡献：**
  - 用 **Gaussian 扰动** 平滑估计 \(\nabla_{\mathbf{A}}\mathcal{J}\)，避免全长轨迹反传；结合 **decoupled**（观测 stop-gradient）降低渲染路径显存。
  - **混合环境：** batch-rendered env 评估策略表现；physics-only env 提供扰动 rollout 做策略改进 → **数量级更少** 的 batch-rendered 并行数（论文表：~10 GB vs PPO 视觉 ~48–50 GB）。
  - **工程：** 自适应探索、reward-invariant 归一化稳定更新。
- **对 wiki 的映射：**
  - [Genesis 仿真器](../../wiki/entities/genesis-sim.md)（官方实现基于 Genesis + Hydra）
  - [Policy Optimization](../../wiki/methods/policy-optimization.md)

### 3) Egocentric 任务套件与 sim2real（§4–5）

- **链接：** <https://github.com/HaoxiangYou/SDPG>
- **核心贡献：**
  - 发布 **egocentric** 视觉 RL benchmark：灵巧操作 + 困难 locomotion；RGB/depth、单/多相机 + 本体感知。
  - **Unitree Go2：** RealSense 深度 egocentric 导航崎岖地形/楼梯；仿真 **<2 h** 单 GPU 训练，**零样本** 上真机。
- **对 wiki 的映射：**
  - [Locomotion](../../wiki/tasks/locomotion.md)
  - [GR00T Visual Sim2Real](../../wiki/entities/gr00t-visual-sim2real.md)（对照：teacher-student 蒸馏 vs 端到端视觉 on-policy）

## BibTeX（项目页）

```bibtex
@misc{you2026efficientonpolicyvisualrlstochastic,
      title={Efficient On-policy Visual-RL via Stochastic Decoupled Policy Gradient},
      author={Haoxiang You and Yilang Liu and Davis Zong and Qian Wang and Teeratham Vitchutripop and Qi Wang and Daniel Rakita and Ian Abraham},
      year={2026},
      eprint={2605.26478},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.26478},
}
```

## 当前提炼状态

- [x] 摘要与项目页方法图对齐
- [x] 项目页核查：**已开源**（GitHub + 安装/训练 README）
- [x] wiki 页面映射确认
