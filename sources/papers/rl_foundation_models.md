# 强化学习基础策略模型（RL Foundation Policy Models）

> Ingest 日期：2026-04-15
> 主题：跨任务通用策略、大规模预训练、机器人基础模型

---

## 核心论文

### Brohan et al. (2022) — RT-1: Robotics Transformer for Real-World Control at Scale
- **核心贡献**：将 Transformer 引入机器人操作控制；在 130k+ 真实机器人演示上训练；1000+ 技能的泛化
- **关键洞见**：规模化数据 + Transformer 架构 → 跨任务泛化；RT-1 开创了"机器人基础模型"方向

### Brohan et al. (2023) — RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
- **核心贡献**：将视觉-语言大模型（PaLM-E）直接输出机器人动作；Web 知识迁移到物理控制
- **关键洞见**：VLA（Vision-Language-Action）模型；语言指令直接驱动低级控制；泛化能力显著超过 RT-1

### Black et al. (2024) — π₀: A Vision-Language-Action Flow Model for General Robot Control
- **核心贡献**：将 Flow Matching 与 VLA 结合；连续动作输出；在 Diffusion Policy 基础上扩展到语言条件
- **关键洞见**：π₀ 统一了 IL、RL 和 VLA 三类方法；Physical Intelligence 的核心模型；适合人形操作任务

### Octo Model Team (2023) — Octo: An Open-Source Generalist Robot Policy
- **核心贡献**：开源通用机器人策略；训练于 Open X-Embodiment 800k 演示数据集；支持多种机器人形态
- **关键洞见**：开源基础模型降低研究门槛；多形态预训练 + fine-tune 范式被广泛采用

### Hansen et al. (2024) — TD-MPC2: Scalable, Robust World Models for Model-Based Control
- **核心贡献**：基于隐空间世界模型的 model-based RL；在 80+ 任务上统一训练单一策略
- **关键洞见**：world model + temporal difference learning = 数据效率与泛化的平衡；适合 locomotion 和操作任务

---

## Wiki 映射

| 论文 / 概念 | 对应 wiki 页面 |
|-----------|--------------|
| RT-1 / RT-2 / π₀ | `wiki/methods/imitation-learning.md` |
| VLA 基础模型 | `wiki/concepts/policy-optimization.md` |
| 通用策略 / Foundation Policy | `wiki/concepts/foundation-policy.md`（新建） |
| Diffusion Policy 扩展 | `wiki/methods/diffusion-policy.md` |
| TD-MPC2 世界模型 | `wiki/methods/model-based-rl.md` |
| Open X-Embodiment 数据集 | `wiki/tasks/locomotion.md` |
| 跨形态迁移 | `wiki/tasks/loco-manipulation.md` |

---

## 关键结论

1. **基础模型范式正在进入机器人控制**：以 RT-1/RT-2/π₀ 为代表，"规模化预训练 + 任务微调"已从 NLP 迁移到机器人领域。
2. **数据规模是瓶颈**：RT-1 需要 130k+ 真实演示；Open X-Embodiment 尝试跨机器人数据共享，但数据质量异构性仍是挑战。
3. **运动控制专用基础模型尚未出现**：目前基础模型主要针对操作任务（pick-place）；locomotion 基础模型（跨地形、跨形态）是下一个前沿。
4. **世界模型是 model-based RL 的希望**：TD-MPC2 / Dreamer 等方法展示了世界模型在样本效率上的优势；与基础模型结合是重要方向。
5. **VLA 架构 = 语言 + 视觉 + 低级动作**：语言作为任务规范，视觉作为状态输入，输出低级关节动作；π₀ 用 Flow Matching 生成连续动作分布，质量优于 Transformer 直接回归。

---

## 参考来源
- RT-1: Brohan et al., arXiv 2022.12
- RT-2: Brohan et al., CoRL 2023
- π₀: Black et al., arXiv 2024.10 (Physical Intelligence)
- Octo: Octo Model Team, arXiv 2023.10
- TD-MPC2: Hansen et al., ICLR 2024
