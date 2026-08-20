# Prism-GRPO（arXiv:2608.17423）

> 来源归档（ingest）

- **标题：** Prism-GRPO: Faster VLA Policy Optimization via Splitting Same-outcome Groups
- **类型：** paper / vla / reinforcement-learning / grpo / sample-efficiency / robotwin
- **arXiv：** <https://arxiv.org/abs/2608.17423>
- **作者：** Zeyun Deng、Yuzhe Lu、Yawei Wang、Linbo Liu、Qing Ping Han、Ding Guande Wu、Panpan Xu、Jun Huan
- **机构：** 亚马逊（Amazon / AWS AI）；真机实验致谢 UCLA Mobility Lab、Purdue DT Lab 实习支持
- **入库日期：** 2026-08-20
- **一句话说明：** 在 SimpleVLA-RL 二值成功 GRPO 上叠加有界轨迹 **execution quality**，把 all-success / all-failure 退化组拆成质量谱，恢复梯度且保持 success 支配。

## 开源状态（步骤 2.5）

- **论文 / 项目：** 无独立项目页；**未发布 Prism-GRPO 专用仓库**。
- **基座：** 明确基于 [SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL) 公开代码与 **OpenVLA-OFT SFT checkpoint**；差异仅在 RL 阶段奖励与 RLOO 优势。
- **结论：** **部分开源** — 可复现栈 = SimpleVLA-RL + 论文算法描述；Prism 补丁本身截至入库日未单独发布。

## 摘录 1：问题（§1、§4）

- **Binary GRPO：** 同场景 G 条 rollout 全成功或全失败 → 优势为 0 → dynamic sampling 丢弃；早期训练浪费大量 rollout。
- **Prism-GRPO：** \(R(\tau)=\mathrm{success}(\tau)+\lambda q(\tau)\)，\(q\in[0,1]\)；成功至少 1、失败至多 \(\lambda\)，**success 仍支配**。
- **理论：** 证明 combined reward **不增加** 获得 informative group 的期望 rollout 数；梯度对齐条件下为 success 的局部上升方向。

**对 wiki 的映射：** 与 [Temporal GRPO](../../wiki/entities/paper-temporal-grpo.md) 对照：后者改 **阶段信用**，Prism 改 **同结果组质量 tie-break**。

## 摘录 2：Quality 信号（Appendix B）

- **接触：** 非目标 peak impulse（默认 GT-Max Force）、contact count、VLM 判碰撞。
- **平滑：** 关节反向计数（Flips / MeanFlips）、action jerk。
- **默认：** \(\lambda=0.2\)；RLOO 优势（非 group-normalized，保留 quality 尺度）。

**对 wiki 的映射：** 强调 task-agnostic 轨迹信号，非 stage progress reward。

## 摘录 3：实验（§5）

- **平台：** RoboTwin 2.0 四任务（Lift Pot、Move Can Pot、Handover Block、Beat Block Hammer）；G=8，512 rollouts/step；OpenVLA-OFT 离散动作头。
- **效率：** 达目标成功率 **rollout 数最多 −56%**（相对 Binary GRPO）。
- **真机：** Piper Move Can Pot 25 trials；Prism **0/25 shove-cheat**（Binary 1/25，RL-ZVP 5/25）；clean success 6/25 vs Binary 4/25。

**对 wiki 的映射：** 链接 [SimpleVLA-RL 开源景观](../../wiki/overview/vla-open-source-repro-landscape-2025.md)。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-prism-grpo.md`**。
- 更新 [VLA 方法页](../../wiki/methods/vla.md)、[manipulation 任务页](../../wiki/tasks/manipulation.md)。
