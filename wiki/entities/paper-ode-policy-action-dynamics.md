---
type: entity
project_id: ode-policy-action-dynamics
tags:
  - paper
  - ntu
  - policy-learning
status: draft
updated: 2026-11-05
arxiv: "2611.01729"
project: https://ntumars.github.io/ODE/
code: https://github.com/NTUMARS/ODE-POLICY
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../methods/reinforcement-learning.md
  - ../overview/embodied-station-2026-11-04-57-papers-technology-map.md
sources:
  - ../../sources/papers/ode_policy_action_dynamics_arxiv_2611_01729.md
  - ../../sources/repos/ode_policy_action_dynamics.md
  - ../../sources/sites/ode-policy-action-dynamics.md
  - ../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md
summary: "ODE Policy 学习动作的物理时间向量场，从上一目标动作积分生成下一目标动作。"
---

# ODE Policy: Robot Learning via Action Dynamics, Not Actions（基于动作动力学的机器人学习）

**ODE Policy**（*ODE Policy: Robot Learning via Action Dynamics, Not Actions*，[arXiv:2611.01729](https://arxiv.org/abs/2611.01729)，[项目页](https://ntumars.github.io/ODE/)，[代码](https://github.com/NTUMARS/ODE-POLICY)）来自 [具身智能小站 2026-11-04 盘点](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)，属「奖励学习与策略后训练」组。

## 一句话定义

**ODE Policy 学习目标动作随物理时间变化的向量场，并从上一个目标动作积分得到后续动作。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ODE | Ordinary Differential Equation | 常微分方程 |
| IL | Imitation Learning | 模仿学习 |
| RL | Reinforcement Learning | 强化学习 |
| VLA | Vision-Language-Action | 视觉语言动作模型 |
| FM | Flow Matching | 流匹配生成式动作头 |
| DiT | Diffusion Transformer | 扩散 Transformer 动作头结构 |
| DDPM | Denoising Diffusion Probabilistic Model | 去噪扩散概率模型 |
| A2A | Action-to-Action Flow Matching | 以历史动作为流起点的流匹配 |
| ACT | Action Chunking with Transformers | 动作块 Transformer 策略 |
| DMP | Dynamic Movement Primitives | 动态运动基元 |
| PPO | Proximal Policy Optimization | 近端策略优化 |

## 核心机制与流程

论文 [arXiv:2611.01729v1](https://arxiv.org/pdf/2611.01729v1) 将向量场作用于发给底层控制器的**目标动作**，并以机器人实际时间积分；这与生成模型中的采样时间不同。图依据论文第 4 节，表示 IL 主流程。

```mermaid
flowchart TB
  obs["图像历史与指令"] --> enc["条件编码"]
  enc --> field["动作向量场"]
  state["本体状态"] --> field
  prev["上一目标动作"] --> field
  field --> integ["物理时间积分"]
  integ --> act["目标动作序列"]
  act --> ctrl["底层控制器"]
  act -. "训练时路径损失" .-> demo["示范动作"]
  act -. "跨动作块延续" .-> prev
```

- IL 在短动作块内积分，用整段示范动作的路径损失反向传播，无需动作导数标签；训练时扰动起点并引导轨迹回归，缓解推理时的误差累积。
- 可选多样性头通过随机化向量场方向表达多模态；RL 场景每个控制步积分一次，可替换 SuperMimic actor。
- 连续的是目标指令轨迹，实际机器人运动还受控制器跟踪误差、采样和硬件限制影响。

## 源码运行时序图

**待完成：**官方 [ODE-POLICY](https://github.com/NTUMARS/ODE-POLICY) 仓库有可运行实现，但相关源码调用链尚未逐项核验。原先的通用推测图已移除；读完源码后补充时序图。README 指出 main 分支含 LIBERO 训练评测及 2D 导航，另有实机和 π0.5 分支。

## 实验与评测

- LIBERO 四套任务上，论文报告视觉运动策略训练 1 epoch 后成功率 79.2%，100 epoch 后 93.8%；π0.5 的 ODE 头在 1 epoch 为 96.0%，原 flow matching 专家为 75.5%。比较须注意后者头部容量和初始化不同。
- 论文报告使用 Euler 积分时生成 8 个动作耗时 5.7 ms，比该实验中的 FM-DiT 快 10.8 倍；这不是所有硬件或求解器的通用倍数。
- 作者报告在 14 项单臂、双臂、灵巧手和人形真机任务上均取得被比较方法中的最高成功率；具体任务、样本量和差异需逐项核对。

## 与其他工作对比

- **实验内对照**：仿真视觉运动 IL 对比六种只换动作头的策略——FM-DiT、FM-UNet、DDPM-DiT、DDPM-UNet、A2A-DiT 与 [ACT](./paper-act.md)，共享图像与语言编码器，编码器外约 24M 参数（§5）。训练 1 epoch 后 ODE Policy 在 LIBERO 四套任务平均成功率 79.2%，基线最高 43.2%（§1、§6.1.1、图 4）；口径为每任务 20 回合、三个随机种子。
- **大模型动作头**：在 π0.5 中用 13.7M 参数、随机初始化的 ODE 头替换 430M 的 flow matching 动作专家（§5）；1 epoch 时 96.0%，随机初始化的 FM 专家为 75.5%，预训练 FM 专家从第 2 epoch 起追平（§6.1.2、图 5A，每任务 10 回合、单种子）。在 [ImageWAM](./paper-rcl-2606-19531-imagewam-do-world-action-models-really-need-vide.md) 中 1 epoch 为 82.6%，原 flow matching 头为 24.6%，A2A 头居中，三者从第 6 epoch 起收敛（图 5B，每任务 25 回合）。各设置回合数与种子数不同，数字不能相互横比。
- **生成时间与物理时间**：[Diffusion Policy](../methods/diffusion-policy.md) 与 flow matching 在无物理意义的生成时间上积分；Streaming Flow Policy 让流时间对齐执行时间，但仍回归由示范导出的速度标签；[Chronos Policy](./paper-chronos-policy-physical-time.md) 把物理时间作为输入，动作仍按固定频率生成（§2.1）。本文在物理时间上积分，只用积分路径监督，不用速度标签，并从上一动作块末端延续。
- **动力系统与动作块衔接**：[DMP](../methods/dmp.md)、稳定动力系统和神经动态策略预先固定向量场形式或收敛到吸引子，每个系统只能表示少数动作（§2.1）。块边界不一致此前靠 ACT 的时间集成或 [Real-Time Chunking](./paper-real-time-chunking.md) 以已提交动作为条件缓解（§2.2）；本文以动作块为积分单元，跨块从上一目标动作继续积分（§4.3）。
- **强化学习**：替换 [SuperMimic](./paper-supermimic.md) 的直接动作 actor，保留其世界模型、critic、奖励和超参（§6.1.3）。在 G1 人形跟踪 40 段 LAFAN1 片段、单 GPU 训练 3 小时的设置下，40 段归一化评测回报为 0.885±0.007，原 actor 为 0.845±0.009，无世界模型的 PPO 为 0.452±0.021（§6.1.3、图 6，三个种子）。

## 结论

**动作动力学提供了连续的目标动作表示，但实际收益与训练设置、求解器、控制接口及任务有关。**

1. 跨块延续的是前一目标动作，不应直接用实测位置替代。
2. 向量场积分不能保证实机状态、力矩或安全约束自动满足。
3. 数字适用于论文对应实验协议，部署前应在目标硬件上测延迟和跟踪质量。
4. 当前为草稿：论文全文和源码调用链尚未全部核验，补齐后才能完成收录。

## 关联页面

- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [embodied-station-2026-11-04-57-papers-technology-map](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)

## 参考来源

- [ode_policy_action_dynamics_arxiv_2611_01729.md](../../sources/papers/ode_policy_action_dynamics_arxiv_2611_01729.md)
- [wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)
- [arXiv:2611.01729](https://arxiv.org/abs/2611.01729)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.01729)
- [项目页](https://ntumars.github.io/ODE/)
- [https://github.com/NTUMARS/ODE-POLICY](https://github.com/NTUMARS/ODE-POLICY)
